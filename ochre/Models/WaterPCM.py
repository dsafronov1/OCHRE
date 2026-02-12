from matplotlib.pylab import f
import numpy as np
import os
import math

from ochre.Models import StratifiedWaterModel

from typing import Dict, Optional

from ochre.utils import schedule





# TODO: priorities: multi-node pcm properties
# TODO: interpolated cp values for non-linear enthalpies
# TODO: find out differences in draw profiles


# PCM properties from manufacturer, same units as water properties
DEFAULT_PCM_PROPERTIES = {
    "h": 600,  # W/m^2K
    "sa_ratio": 15, # m^2/m^3 of total pcm heat exchanger volume
    "h_conv": 100,  # W/m^2-K, accounts for surface area (ha)
    "solid": {
        "pcm_density": 0.904,  # g/cm^3
        "pcm_cp": 0.6,  # J/g-C # adjusted by real measurements average from 0-45c
        "pcm_conductivity":0.2,  # W/m-C
    },
    "enthalpy_lut_file": "cp_h-T_data_shifted_120F.csv",
}

def _normalize_pcm_properties(pcm_properties: Optional[Dict]) -> Dict:
    if pcm_properties is None:
        pcm_properties = {}

    props = {k: v for k, v in pcm_properties.items() if k not in ("solid", "enthalpy_lut")}
    solid = pcm_properties.get("solid", {})
    normalized = {**DEFAULT_PCM_PROPERTIES, **props}
    normalized["solid"] = {**DEFAULT_PCM_PROPERTIES["solid"], **solid}
    return normalized

def _resolve_enthalpy_lut_file(pcm_properties: Dict) -> str:
    pcm_file = pcm_properties.get("enthalpy_lut_file") or DEFAULT_PCM_PROPERTIES["enthalpy_lut_file"]
    if isinstance(pcm_file, os.PathLike):
        pcm_file = os.fspath(pcm_file)
    if not isinstance(pcm_file, str):
        raise TypeError("enthalpy_lut_file must be a filename string.")
    return pcm_file

def _load_enthalpy_lut(pcm_file: str) -> np.ndarray:
    full_path = os.path.join(os.path.dirname(__file__), f"../defaults/pcm_configs/{pcm_file}")
    return np.loadtxt(full_path, delimiter=",", skiprows=1)

def calculate_interpolation_data(enthalpy_lut):
    temps = enthalpy_lut[:, 0].astype(float)
    # Convert all values if any appear to be in Kelvin
    if np.any(temps > 273.15):
        temps = temps - 273.15

    specific_heats = enthalpy_lut[:, 1].astype(float)  # J/g
    enthalpies = enthalpy_lut[:, 2].astype(float)      # J/g
    return temps, specific_heats, enthalpies




def get_pcm_enthalpy(t_pcm, enthalpy_lut):
    '''look up the enthalpy of the PCM using an interpolated LUT'''
    # col 1 : t_pcm (C), col 2 : cp (J/(gC)), col 3 : enthalpy (J/kg)
    # must be in ascending sorted order by t_pcm
    if t_pcm < enthalpy_lut[0,0] or t_pcm > enthalpy_lut[-1,0]:
        raise ValueError(f"t_pcm {t_pcm} is outside the range of the LUT [{enthalpy_lut[0,0]} to {enthalpy_lut[-1,0]}]")
    
    idx = (enthalpy_lut[:,0] <= t_pcm).nonzero()[0][-1]
    
    t_low, t_high = enthalpy_lut[idx:idx+2, 0]
    h_low, h_high = enthalpy_lut[idx:idx+2, 2]
    
    return h_low + (t_pcm - t_low) * (h_high - h_low) / (t_high - t_low)

class TankWithPCM(StratifiedWaterModel):
    """
    Water Tank Model with Phase Change Material

    Defaults to a 13 node tank with 12 water nodes and 1 PCM node.
    """

    name = "Water Tank with PCM"

    def __init__(self, pcm_properties=DEFAULT_PCM_PROPERTIES, pcm_water_node=5, pcm_vol_fraction=0.5, **kwargs):
        
        # PCM node data
        self.pcm_water_node = pcm_water_node  # node number, from the top
        self.t_pcm_wh_idx = self.pcm_water_node - 1
        self.pcm_vol_fraction = pcm_vol_fraction
        self.pcm_mass = None  # in g
        
        super().__init__(**kwargs)
        
        self.pcm_properties = _normalize_pcm_properties(pcm_properties)
        pcm_file = _resolve_enthalpy_lut_file(self.pcm_properties)
        self.pcm_properties["enthalpy_lut_file"] = pcm_file
        self.enthalpy_lut = _load_enthalpy_lut(pcm_file)

        # Bounds check for pcm_vol_fraction for stability
        if not (6.582730627258115e-08 <= self.pcm_vol_fraction <= 0.9999999999999725):
            raise ValueError(f"pcm_vol_fraction {pcm_vol_fraction} must be between (6.582730627258115e-08 and 0.9999999999999725) to ensure stability.")
        if self.pcm_vol_fraction < 0.01 or self.pcm_vol_fraction > 0.99:
            self.warn(f"pcm_vol_fraction {pcm_vol_fraction} is outside the recommended range (0.01 to 0.99). Results may be inaccurate.")
        
        self.key_temp, self.specific_heats, self.key_enthalpy = calculate_interpolation_data(self.enthalpy_lut)
        self.key_enthalpy *= self.pcm_mass  # in J

        # PCM state and input indices
        self.t_pcm_idx = self.state_names.index("T_PCM")
        self.h_pcm_idx = self.input_names.index("H_PCM")
        assert self.state_names.index(f"T_WH{self.pcm_water_node}") == self.t_pcm_wh_idx
        self.h_pcm_wh_idx = self.input_names.index(f"H_WH{self.pcm_water_node}")

        # PCM results variables
        self.pcm_heat_to_water = None  # in W
        t_pcm = self.states[self.t_pcm_idx]  # PCM temperature, in C
        self.enthalpy_pcm = np.interp(t_pcm, self.key_temp, self.key_enthalpy)  # PCM enthalpy, in J

    def load_rc_data(self, **kwargs):
        rc_params = super().load_rc_data(**kwargs)
#
        # Add PCM capacitance, default to solid state for now
        pcm_node_vol_fraction = self.vol_fractions[self.t_pcm_wh_idx]
        pcm_volume = self.volume * pcm_node_vol_fraction * self.pcm_vol_fraction
        self.pcm_mass = self.pcm_porperties["solid"]["pcm_density"] * pcm_volume / 1e3  # in g
        rc_params["C_PCM"] = self.pcm_porperties["solid"]["pcm_cp"] * self.pcm_mass  # in J/K
        # rc_params["C_PCM"] = PCM_PROPERTIES["solid"]["pcm_c"] * pcm_volume

        # Reduce water volume and vol_fractions from PCM volume
        self.volume -= pcm_volume
        self.vol_fractions[self.t_pcm_wh_idx] *= 1 - self.pcm_vol_fraction
        self.vol_fractions = self.vol_fractions / sum(self.vol_fractions)

        # Reduce water node capacitance
        rc_params[f"C_WH{self.pcm_water_node}"] *= 1 - self.pcm_vol_fraction

        # Add water-PCM resistance, default to solid state for now rc_params[f"R_PCM_WH{self.pcm_water_node}"] = 1 / PCM_PROPERTIES["h_conv"]
       
#
        return rc_params

    def get_pcm_heat_xfer(self):
        # if convection coefficient changes by phase, add heat transfer here

        # # calculate heat transfer (pcm to water)
        # t_water = self.states[self.t_pcm_wh_idx]
        # t_pcm = self.states[self.t_pcm_idx]
        # h_pcm = PCM_PROPERTIES["h_conv"] * (t_pcm - t_water)  # in W
        # return h_pcm
        return 0 # keep zero for now to prevent double counting pcm heat transfer


    def update_inputs(self, schedule_inputs=None):
        # Note: self.inputs_init are not updated here, only self.current_schedule
        super().update_inputs(schedule_inputs)

        # get heat injections from PCM
        self.pcm_heat_to_water = self.get_pcm_heat_xfer()

        # add PCM heat to inputs       
        self.inputs_init = np.append(self.inputs_init, -self.pcm_heat_to_water) # fix heat flow direction
        # self.inputs_init[self.h_pcm_idx] = self.pcm_heat_to_water
        self.inputs_init[self.h_pcm_wh_idx] += self.pcm_heat_to_water # fix heat flow direction

    def update_model(self, control_signal=None):
        super().update_model(control_signal)

        # calculate new PCM enthalpy based on linear temperature change
        delta_t = self.next_states[self.t_pcm_idx] - self.states[self.t_pcm_idx]
        q_pcm = delta_t * self.capacitances[self.t_pcm_idx]
        self.enthalpy_pcm += q_pcm

        # update PCM temperature in new_states
        t_pcm = np.interp(self.enthalpy_pcm, self.key_enthalpy, self.key_temp)
        self.next_states[self.t_pcm_idx] = t_pcm

    def generate_results(self):
        # Note: most results are included in Dwelling/WH. Only inputs and states are saved to self.results
        results = super().generate_results()

        if self.verbosity >= 6:
            results["Water Tank PCM Temperature (C)"] = self.states[self.t_pcm_idx]
            results["Water Tank PCM Water Temperature (C)"] = self.states[self.t_pcm_wh_idx]
            results["Water Tank PCM Enthalpy (J)"] = self.enthalpy_pcm
            results["Water Tank PCM Heat Injected (W)"] = self.pcm_heat_to_water
        return results
    
    
    

    
    
class TankWithMultiPCM(StratifiedWaterModel):
    """
    Water Tank Model with Phase Change Material in Multiple Nodes
    
    Input a dictionary of {pcm_node: pcm_vol_fractions} to select the nodes and vol_fractions the pcm will be in

    Defaults to a 15 node tank with 12 water nodes and 3 PCM node.
    """

    name = "Water Tank with Internal Multi PCM"

    def __init__(self, pcm_properties: dict, pcm_node_vol_fractions: dict[int:float]={4:0.5, 5:0.5, 6:0.5}, **kwargs):
        
        # PCM node data
        self.pcm_node_vol_fractions = pcm_node_vol_fractions  # node number, from the top
        self.pcm_water_nodes = list(self.pcm_node_vol_fractions.keys())
        self.t_pcm_wh_idx = [node -1 for node in self.pcm_water_nodes]
        self.pcm_vol_fraction = list(self.pcm_node_vol_fractions.values())
        self.pcm_mass_kg = None  # in g
        self.external_nodes = ['AMB']
        self.pcm_heat_to_water_rc_network = None
        self.enthalpy_pcm = None
        self.pcm_properties = _normalize_pcm_properties(pcm_properties)
        pcm_file = _resolve_enthalpy_lut_file(self.pcm_properties)
        self.pcm_properties["enthalpy_lut_file"] = pcm_file
        self.enthalpy_lut = _load_enthalpy_lut(pcm_file)
        self.h = self.pcm_properties['h']
        self.ha = self.pcm_properties['h_conv']
        self.conductivity= self.pcm_properties['solid']['pcm_conductivity']
        self.key_temp, self.key_specific_heats, self.key_enthalpy = calculate_interpolation_data(self.enthalpy_lut)
        
        print(f"PCM enthalpy LUT file: {pcm_file} for case ha:{self.ha:.2f} with sa_ratio:{self.pcm_properties['sa_ratio']:.2f}")

        # pcm_cp cache between
        self._pcm_cp_cache: dict[tuple[float, ...], np.ndarray] = {}
        
        super().__init__(**kwargs)
        # self.n_nodes = len(self.output_names)

        # Bounds check for pcm_vol_fraction for stability
        # for node, vol_fraction in self.pcm_node_vol_fractions.items():
        #     if not (6.582730627258115e-08 <= vol_fraction <= 0.9999999999999725):
        #         raise ValueError(f"pcm_node: {node} vol_fraction {vol_fraction} must be between 6.582730627258115e-08 and 0.9999999999999725 to ensure stability.")
        #     if vol_fraction < 0.01 or vol_fraction> 0.85:
        #         self.warn(f"pcm_node: {node} pcm_vol_fraction {vol_fraction} is outside the recommended range (0.01 to 0.99). Results may be inaccurate.")
        # self.time_res = datetime.timedelta(seconds=5)

        # PCM state and input indices
        self.t_pcm_idx = [i for i, name in enumerate(self.state_names) if "T_PCM" in name]
        self.h_pcm_idx = [i for i, name in enumerate(self.input_names) if "H_PCM" in name]
        assert [self.state_names.index(f"T_WH{node}") for node in self.pcm_water_nodes] == self.t_pcm_wh_idx
        self.h_pcm_wh_idx = [self.input_names.index(f"H_WH{node}") for node in self.pcm_water_nodes]
        
        # PCM results variables
        self.pcm_heat_to_water = None  # in W
        # t_pcm = self.states[self.t_pcm_idx]  # PCM temperature, in C
        # self.enthalpy_pcm = np.interp(t_pcm, self.key_temp, self.key_enthalpy) # PCM enthalpy, in J
        self.enthalpy_pcm: np.ndarray = np.interp(
            self.states[self.t_pcm_idx], self.key_temp, self.key_enthalpy * self.pcm_mass_kg * 1e3
        )

    def load_rc_data(self, **kwargs):
        rc_params = super().load_rc_data(**kwargs)
        self.rc_params = rc_params
        
        # Create a dictionary to store PCM mass, capacitance, and resitances for each node
        self.pcm_mass_dict = {}
        self.pcm_node_properties = {}
        c_pcm_dict = {}
        r_wh_pcm_dict = {}
        r_pcm_pcm_dict = {}
        r_pcm_amb_dict = {}
        total_pcm_volume = 0.0
        total_pcm_mass = 0.0
        original_volume = self.volume  # Keep the original volume for proper PCM volume computation
        
        # parameters to calcualte pcm-pcm conductivity resistance
        # water heater height 4ft (static)
        # backout radius from volume and height
        
        self.tank_height = 4 * 0.3048 # in m
        start_temp = kwargs.get('Setpoint Temperature (C)', 51.66666666666667)
        self.volume_m3 = self.volume / 1e3 # convert liters to m3
        self.tank_radius = math.sqrt(self.volume_m3 / (math.pi * self.tank_height))
        effective_area_ratio = 0.8
        length = self.tank_height / 12 # length on 1 cell
        water_heater_cross_area = math.pi * self.tank_radius**2
        effective_area = effective_area_ratio * water_heater_cross_area

        # Loop over each PCM node and apply the modifications
        for i, node in enumerate(self.pcm_water_nodes):
            # Convert the 1-indexed node number to a 0-indexed index
            idx = node - 1
            
            # Get the PCM volume fraction for this node
            pcm_frac = self.pcm_node_vol_fractions[node]
            
            # Calculate the water volume present in this node (before PCM extraction)
            # volume is in Liters
            node_volume = original_volume * self.vol_fractions[idx]
            
            # Compute the PCM volume in this node
            pcm_volume = node_volume * pcm_frac
            total_pcm_volume += pcm_volume
            
            # Compute the PCM mass in this node (in grams) and store it
            pcm_mass = self.pcm_properties["solid"]["pcm_density"] * pcm_volume * 1e3 # ensure units are in g
            self.pcm_mass_dict[node] = pcm_mass
            total_pcm_mass += pcm_mass
            
            # Add PCM capacitance for this node (in J/K)
            pcm_cp = np.interp(start_temp, self.enthalpy_lut[:,0], self.enthalpy_lut[:,1])
            c_pcm_dict[f"C_PCM{node}"] = pcm_cp * pcm_mass
            # c_pcm_dict[f"C_PCM{node}"] = 1e-3
            
            # Reduce the water node capacitance for this node by the PCM fraction
            rc_params[f"C_WH{node}"] *= (1 - pcm_frac)
            
            # Update the water node’s volume fraction to remove the PCM volume fraction
            self.vol_fractions[idx] *= (1 - pcm_frac)
            
            # Add the water-PCM thermal resistance for this node (in K/W)
            ha = self.pcm_properties["h"] * pcm_volume * 1e-3 * self.pcm_properties["sa_ratio"] # in W/K
            r_wh_pcm_dict[f"R_PCM{node}_WH{node}"] = 1 / ha
            
            # use pure thermal conductivity for this
            if i < len(self.pcm_water_nodes) - 1:
                next_pcm_node = self.pcm_water_nodes[i + 1]
                if next_pcm_node - node == 1:  # Check if sequential
                    r_pcm_pcm_dict[f"R_PCM{node}_PCM{next_pcm_node}"] = length/(self.pcm_properties['solid']['pcm_conductivity'] * effective_area * pcm_frac) # L/(kA)
                    
            # add PCM-AMB thermal resistance for node (in K/W) Should be arbitrarily high
            r_pcm_amb_dict[f"R_PCM{node}_AMB"] = 200000
            self.pcm_node_properties[node] = {"volume[L]": pcm_volume, "ha": ha, "sa_ratio": self.pcm_properties["sa_ratio"], "h": self.pcm_properties["h"]}
            print(f"Node {node} pcm mass: {(pcm_mass)/1000:.3e} kg")        
        # extend the rc_params dictionary with the thermal capacitance and resitances
        rc_params.update(c_pcm_dict)
        rc_params.update(r_wh_pcm_dict)
        rc_params.update(r_pcm_amb_dict)
        rc_params.update(r_pcm_pcm_dict)


        
        # Subtract the total PCM volume from the global water volume
        # TODO - add in option to do external pcm volumes
        self.volume -= total_pcm_volume
        self.pcm_mass_kg = total_pcm_mass / 1000
        print(f"Total PCM mass: {(self.pcm_mass_kg):.3e} kg")
        
        # Normalize the water volume fractions so that they sum to 1
        self.vol_fractions = self.vol_fractions / np.sum(self.vol_fractions)
        
        self.rc_params = rc_params
        return rc_params
    
    def update_rc_network(self, t_pcm, **kwargs):
        '''Get the dynamic specific heat for each of the pcm nodes and update the capacitance in the rc_network'''
        cache_key = tuple(np.round(t_pcm, 3))
        pcm_specific_heats = self._pcm_cp_cache.get(cache_key)
        if pcm_specific_heats is None:
            pcm_specific_heats = np.interp(t_pcm, self.key_temp, self.key_specific_heats)
            self._pcm_cp_cache[cache_key] = pcm_specific_heats


        # Loop over each PCM node and apply the modifications
        for i, node in enumerate(self.pcm_water_nodes):
            
            pcm_node_mass = self.pcm_mass_dict[node]
            pcm_specific_heat = pcm_specific_heats[i]

            
            # Update the capacitance for this node (in J/K)
            self.rc_params[f"C_PCM{node}"] = pcm_specific_heat * pcm_node_mass
            # self.rc_params[f"C_PCM{node}"] = 1e-3
        return self.rc_params
        
    def update_state_space_model(self, **kwargs):
        
        pcm_temps = self.next_states[self.t_pcm_idx]
        dynamic_rc_params = self.update_rc_network(pcm_temps)
        all_cap = {name.upper().split('_')[1:][0]: val 
               for name, val in dynamic_rc_params.items() if name[0] == 'C'}
        all_res = {tuple(name.upper().split('_')[1:]): val 
                for name, val in dynamic_rc_params.items() if name[0] == 'R'}
        
        # You may need to re-identify internal/external nodes if not stored already
        internal_nodes = [node for node in all_cap.keys()]
        
        external_nodes = [node for node in self.external_nodes]  # assuming these were stored
        
        # Recompute the state-space matrices
        A_c, B_c = self.create_rc_matrices(all_cap, all_res, internal_nodes, external_nodes)
        
        # Update the model’s matrices
        self.A = A_c
        self.B = B_c
        self.capacitances = np.array(list(all_cap.values()))
        
        # Define state and input names
        state_names = ['T_' + node for node in internal_nodes]
        input_names = ['T_' + node for node in external_nodes] + ['H_' + node for node in internal_nodes]
        
        outputs = None
        matrices=(A_c, B_c)
        
        # if unused_inputs is not None:
        #     good_input_idx = [i for (i, name) in enumerate(input_names) if name not in unused_inputs]
        #     B_c = B_c[:, good_input_idx]
        #     input_names = [name for name in input_names if name not in unused_inputs]

        # initialize states based on matrices
        
        # Define states
        self.nx = len(self.states)
        # if isinstance(self.states, dict):
        #     self.states = np.array(list(self.states.values()), dtype=float)
        # else:
        #     self.states = np.zeros(self.nx, dtype=float)

        # Define inputs
        self.nu = len(input_names)
        # if isinstance(input_names, dict):
        #     self.inputs = np.array(list(input_names.values()), dtype=float)
        # else:
        #     self.inputs = np.zeros(self.nu, dtype=float)
        self.input_names = list(input_names)
        self.use_schedule_for_inputs = all([col in self.input_names for col in self.schedule.columns])
        self.inputs_init = self.inputs  # for saving values from update_inputs step
        
        # Define outputs
        if outputs is None:
            self.ny = self.nx
            self.output_names = self.state_names.copy()
        else:
            self.ny = len(outputs)
            self.output_names = outputs
        # self.outputs = np.zeros(self.ny, dtype=float)
        # self.next_outputs = self.outputs  # for saving outputs of next time step

        # Define continuous-time matrices
        self.A_c, self.B_c, self.C, self.D = self.create_matrices(matrices)

        # Update output values
        self.outputs = self.C.dot(self.states) + self.D.dot(self.inputs)

        # Reduce model order (i.e. number of states)
        self.reduced = False
        self.transformation_matrix = None
        if 'reduced_states' in kwargs or 'reduced_min_accuracy' in kwargs:
            self.reduce_model(update_discrete=False, **kwargs)

        # Create A, B discrete matrices
        self.A, self.B = self.to_discrete()


    def get_pcm_heat_xfer(self):
        # if convection coefficient changes by phase, add heat transfer here


        return np.array([0] * len(self.t_pcm_idx))

        # # calculate heat transfer (pcm to water)
        # t_water = self.states[self.t_pcm_wh_idx]
        # t_pcm = self.states[self.t_pcm_idx]
        # h_pcm = PCM_PROPERTIES["h_conv"] * (t_pcm - t_water)  # in W
        # return h_pcml
        return 0 # keep zero for now to prevent double counting pcm heat transfer


    def update_inputs(self, schedule_inputs=None):
        # Note: self.inputs_init are not updated here, only self.current_schedule
        super().update_inputs(schedule_inputs)

        # get heat injections from PCM
        self.pcm_heat_to_water = self.get_pcm_heat_xfer()

        # add PCM heat to inputs       
        self.inputs_init = np.append(self.inputs_init, -self.pcm_heat_to_water) # fix heat flow direction
        # self.inputs_init[self.h_pcm_idx] = self.pcm_heat_to_water
        self.inputs_init[self.h_pcm_wh_idx] += self.pcm_heat_to_water # fix heat flow direction

    # def update_model(self, control_signal=None):
    #     super().update_model(control_signal)

    #     # calculate new PCM enthalpy based on linear temperature change
    #     delta_t = self.next_states[self.t_pcm_idx] - self.states[self.t_pcm_idx] 
    #     q_pcm = delta_t * self.capacitances[self.t_pcm_idx] # J
    #     self.pcm_heat_to_water_rc_network = -q_pcm/self.time_res.total_seconds()
    #     self.enthalpy_pcm += q_pcm

    #     # update PCM temperature in new_states
    #     t_pcm = np.interp(self.enthalpy_pcm, self.key_enthalpy, self.key_temp)
    #     self.next_states[self.t_pcm_idx] = t_pcm
    #     self.update_state_space_model()
        
    def update_model(self, control_signal=None, epsilon=None, max_iter=None, iter_count=0,
                    original_states=None, original_inputs=None, original_inputs_init=None):
        # Use provided convergence criteria or fall back to instance attributes.
        # if epsilon is None:
        #     epsilon = self.epsilon
        # if max_iter is None:
        #     max_iter = self.max_iter

        # # On the first call, store copies of the original states, inputs, and inputs_init.
        # if original_states is None:
        #     original_states = self.states.copy()
        # if original_inputs is None:
        #     original_inputs = self.inputs.copy()
        # if original_inputs_init is None:
        #     original_inputs_init = self.inputs_init.copy()

        # Call the parent's update_model function.
        super().update_model(control_signal)
        
        # Only use temperatures output from state space model and enthalpies nothing else
        masses = np.array([self.pcm_mass_dict[node] for node in self.pcm_water_nodes]) # in gs
        enthalpy_state = np.interp(self.states[self.t_pcm_idx], self.key_temp, self.key_enthalpy) * masses
        enthalpy_next_state = np.interp(self.next_states[self.t_pcm_idx], self.key_temp, self.key_enthalpy) * masses
        q_pcm = enthalpy_next_state - enthalpy_state
        self.pcm_heat_to_water_rc_network = -q_pcm / self.time_res.total_seconds()
        self.enthalpy_pcm = enthalpy_next_state
        self.delta_enthalpy_pcm = enthalpy_next_state - enthalpy_state
        

        # Update the PCM temperature using interpolation from enthalpy to temperature.
        # t_pcm = np.interp(self.enthalpy_pcm, self.key_enthalpy, self.key_temp)
        # t_pcm = self.next_states[self.t_pcm_idx]
        # self.next_states[self.t_pcm_idx] = t_pcm

        # Call the state space model update (which may modify states, inputs, and inputs_init).
        self.update_state_space_model()

        # Reset the states and inputs to their original values so they stay constant between iterations.
        # self.states = original_states.copy()
        # self.inputs = original_inputs.copy()
        # self.inputs_init = original_inputs_init.copy()

        # # Check convergence: if the maximum change is smaller than epsilon or max iterations reached, stop.
        # if np.max(np.abs(delta_t)) < epsilon or iter_count > max_iter:
        #     self.step_num += 1
        #     print(f"Convergence reached after {iter_count + 1} iterations: ΔT = {np.max(np.abs(delta_t)):.6f} [{self.step_num}/{self.sim_times.size}]")
        #     return
        # else:
        #     # Update the current state for the next iteration.
        #     self.states[self.t_pcm_idx] = t_pcm

        #     # Recursively call update_model with the original copies maintained.
        #     self.update_model(control_signal, epsilon, max_iter, iter_count + 1,
        #                     original_states, original_inputs, original_inputs_init)
        
    def generate_results(self):
        # Note: most results are included in Dwelling/WH. Only inputs and states are saved to self.results
        results = super().generate_results()

        if self.verbosity >= 3:
            results['Total PCM Enthalpy (J)'] = self.enthalpy_pcm.sum()
            results['Delta Total PCM Enthalpy (J)'] = self.delta_enthalpy_pcm.sum()
            results['Total PCM Heat Injected (W)'] = self.pcm_heat_to_water_rc_network.sum()

        if self.verbosity >= 6:
            
            for i, idx in enumerate(self.t_pcm_idx):
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Temperature (C)"] = self.states[idx]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Water Temperature (C)"] = self.states[self.t_pcm_wh_idx[i]]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Enthalpy (J)"] = self.enthalpy_pcm[i]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Enthalpy Delta (J)"] = self.delta_enthalpy_pcm[i]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Heat Injected (W)"] = self.pcm_heat_to_water_rc_network[i]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Capacitance (J/K)"] = self.capacitances[idx]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} h (W/m^2K)"] = self.pcm_node_properties[self.t_pcm_wh_idx[i]+1]['h']
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} volume (L)"] = self.pcm_node_properties[self.t_pcm_wh_idx[i]+1]['volume[L]']
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} sa_ratio"] = self.pcm_node_properties[self.t_pcm_wh_idx[i]+1]['sa_ratio']
                
            
            results['PCM Mass (kg)'] = self.pcm_mass_kg
            results['Water Volume (L)'] = self.volume

            
        return results
    
    
    
    
class TankWithMultiPCMExternal(StratifiedWaterModel):

    """Stratified electric water-heater with a concentric external PCM layer."""

    name = "Water Tank with External PCM"
    def __init__(
        self,
        pcm_properties: Dict,
        *,
        pcm_thickness_in: float = 1,
        pcm_split_thickness_in: float = 0.005,     # in PCM thickness
        insulation_thickness_in: float = 2,       # in insulation thickness
        insulation_k_value: float = 0.0484,         # W/m·K
        insulation_cp_value: float = 1000.0,      # J/kg·K
        # insulation_cp_value: float = 1e-2,      # J/kg·K
        insulation_density: float = 40.0,         # kg/m³
        enamel_thickness_in: float = 0.008,       # ≈0.2 mm glass-enamel
        enamel_k_value: float = 1.0,              # W/m·K (vitreous enamel)
        enamel_cp_value: float = 840.0,           # J/kg·K (glass)
        # enamel_cp_value: float = 1e-2,           # J/kg·K (glass)
        enamel_density: float = 2500.0,           # kg/m³ (glass)
        steel_wall_thickness_in: float = 0.1,      # ≈2.75 mm total wall
        steel_k_value: float = 55.0,              # W/m·K mildsteel
        steel_cp_value: float = 490.0,            # J/kg·K mild steel
        # steel_cp_value: float = 1e-2,            # J/kg·K mild steel
        steel_density: float = 7850.0,            # kg/m³
        water_side_film_h: float | None = 50,    # W/m²·K
        **kwargs,
    ) -> None:
        """Parameters
        ----------
        pcm_thickness_in: float The thickness of the PCM external to the tank in inches
        """
        IN_TO_M = 0.0254
        # Store material properties
        self.pcm_properties = _normalize_pcm_properties(pcm_properties)
        self.insulation_k_value = insulation_k_value
        self.insulation_cp_value = insulation_cp_value
        self.insulation_density = insulation_density
        self.enamel_k_value = enamel_k_value
        self.enamel_cp_value = enamel_cp_value
        self.enamel_density = enamel_density
        self.steel_k_value = steel_k_value
        self.steel_cp_value = steel_cp_value
        self.steel_density = steel_density
        

        # ------------------------------------------------------------------
        # Pre‑load PCM enthalpy lookup table
        pcm_file = _resolve_enthalpy_lut_file(self.pcm_properties)
        self.pcm_properties["enthalpy_lut_file"] = pcm_file
        self.enthalpy_lut = _load_enthalpy_lut(pcm_file)

        # Fallback for water‑side film‑coefficient
        self.water_side_film_h = self.pcm_properties.get('film_h')
        self.pcm_thickness_in = self.pcm_properties.get('external_pcm_thickness_in')
        self.pcm_segment_thickness_inches = self.pcm_properties.get("pcm_segment_thickness_inches", 0.2)
        self.external_nodes = ['AMB']

        # ------------------------------------------------------------------
        # Geometry needed *before* parent init so that volumes are correct
        self.tank_height_m = kwargs.get("tank_height_m", 4 * 0.3048)   # 4 ft default
        self.volume_L = kwargs.get("volume", 136.275)                   # 40 gal tank default
        self.volume_m3 = self.volume_L / 1e3
        self.tank_radius_m = math.sqrt(self.volume_m3 / (math.pi * self.tank_height_m))
        self.internal_area_m2 = 2 * math.pi * self.tank_radius_m * self.tank_height_m
        self.top_area_m2 = math.pi * self.tank_radius_m**2
        self.shell_area_m2 = self.internal_area_m2 + 2 * self.top_area_m2

        # Convert thicknesses to meters
        self.enamel_thickness_m = float(enamel_thickness_in * IN_TO_M)
        self.steel_wall_thickness_m = float(steel_wall_thickness_in * IN_TO_M)
        self.pcm_thickness_m = float(self.pcm_thickness_in * IN_TO_M)
        # self.insulation_thickness_m = float(insulation_thickness_in * IN_TO_M)
        # max 2 inches of pcm + insulation
        self.insulation_thickness_m = float(2.0 * IN_TO_M - self.pcm_thickness_m)

        # ------------------------------------------------------------------
        # Compute U-values for each layer
        # Enamel layer
        enamel_inner_r = self.tank_radius_m
        enamel_mean_r = enamel_inner_r + self.enamel_thickness_m / 2
        self.enamel_u_value_W_per_m2_K = (
            self.enamel_k_value
            / (enamel_mean_r * math.log((enamel_inner_r + self.enamel_thickness_m) / enamel_inner_r))
        )
        # Steel wall layer
        steel_inner_r = self.tank_radius_m - self.enamel_thickness_m
        steel_mean_r = steel_inner_r + self.steel_wall_thickness_m / 2
        self.steel_u_value_W_per_m2_K = (
            self.steel_k_value
            / (steel_mean_r * math.log((steel_inner_r + self.steel_wall_thickness_m) / steel_inner_r))
        )
        # Insulation layer
        ins_inner_r = self.tank_radius_m + self.pcm_thickness_m
        ins_mean_r = ins_inner_r + self.insulation_thickness_m / 2
        self.insulation_u_value_W_per_m2_K = (
            self.insulation_k_value
            / (ins_mean_r * math.log((ins_inner_r + self.insulation_thickness_m) / ins_mean_r))
        )

        # --------------------------------------------------------------- parent
        super().__init__(**kwargs)

        # -------------------------------------------------------------- post-init

        self.key_temp, self.key_specific_heats, self.key_enthalpy = calculate_interpolation_data(self.enthalpy_lut)
        # self.key_enthalpy *= (self.pcm_mass_kg * 1e3)
        
        
        # Dynamic state bookkeeping
        self.t_pcm_idx = [i for i, name in enumerate(self.state_names) if "T_PCM" in name]
        self.h_pcm_idx = [i for i, name in enumerate(self.input_names) if "H_PCM" in name]
        self.t_wh_idx = [i for i, name in enumerate(self.state_names) if "T_WH" in name]
        self.t_en_idx = [i for i, name in enumerate(self.state_names) if "T_ENM" in name]
        # assert [self.state_names.index(f"T_WH{node}") for node in self.pcm_water_nodes] == self.t_pcm_wh_idx
        # self.h_pcm_wh_idx = [self.input_names.index(f"H_WH{node}") for node in self.pcm_water_nodes]
        
        self.t_pcm_wh_idx = [name for name in self.state_names if name.startswith("T_PCM")]
        
        self.enthalpy_pcm: np.ndarray = np.interp(
            self.states[self.t_pcm_idx], self.key_temp, self.key_enthalpy * self.pcm_mass_kg * 1e3
        )
        
    def load_rc_data(self, **kwargs):
        include_axial = kwargs.get("include_axial_conduction", True)
        rc = super().load_rc_data(**kwargs)

        n_nodes = len(self.vol_fractions)
        A_layer = self.internal_area_m2 / n_nodes
        L_layer = self.tank_height_m / n_nodes
        start_T = kwargs.get("Setpoint Temperature (C)", 51.6666667)

        # material props
        rho_en = kwargs.get("rho_enamel", self.enamel_density)
        c_en   = kwargs.get("c_enamel",   self.enamel_cp_value)
        rho_st = kwargs.get("rho_steel",  self.steel_density)
        c_st   = kwargs.get("c_steel",    self.steel_cp_value)
        rho_ins= kwargs.get("rho_ins",    self.insulation_density)
        c_ins  = kwargs.get("c_ins",      self.insulation_cp_value)
        k_en   = kwargs.get("k_enamel",   self.enamel_k_value)
        k_st   = kwargs.get("k_steel",    self.steel_k_value)
        k_pcm  = self.pcm_properties["solid"]["pcm_conductivity"]
        k_ins  = kwargs.get("insulation_k_value", self.insulation_k_value)
        h_ext  = kwargs.get("h_ext",       8.0)

        # radii for layers
        r_w   = self.tank_radius_m
        r_en  = r_w + self.enamel_thickness_m
        r_st  = r_en + self.steel_wall_thickness_m
        r_pcm = r_st + self.pcm_thickness_m
        r_ins = r_pcm + self.insulation_thickness_m

        # mid radii for axial conduction areas
        r_mid_en  = r_w + 0.5*self.enamel_thickness_m
        r_mid_st  = r_en + 0.5*self.steel_wall_thickness_m
        r_mid_pcm = r_st + 0.5*self.pcm_thickness_m
        r_mid_ins = r_pcm + 0.5*self.insulation_thickness_m

        A_vert_en  = 2*math.pi*r_mid_en * self.enamel_thickness_m
        A_vert_st  = 2*math.pi*r_mid_st * self.steel_wall_thickness_m
        A_vert_pcm = 2*math.pi*r_mid_pcm* self.pcm_thickness_m
        A_vert_ins = 2*math.pi*r_mid_ins* self.insulation_thickness_m

        # PCM layer thickness parameters (adjustable from 0.005" to 20")
        pcm_in_max = 20
        pcm_in_min = 1e-12
        IN_TO_M = 0.0254
        if self.pcm_thickness_m/IN_TO_M < pcm_in_min:
            print(f"Warning: PCM thickness {self.pcm_thickness_m/IN_TO_M} inches is less than {pcm_in_min} inches. Clamping to {pcm_in_min} inches.")
            pcm_thickness_inches = pcm_in_min
        elif self.pcm_thickness_m/IN_TO_M > pcm_in_max:
            print(f"Warning: PCM thickness {self.pcm_thickness_m/IN_TO_M} is greater than {pcm_in_max} inches. Clamping to {pcm_in_max} inches.")
            pcm_thickness_inches = pcm_in_max
        else:
            pcm_thickness_inches = kwargs.get("pcm_thickness_inches", self.pcm_thickness_m / IN_TO_M)
            pcm_thickness_inches = max(pcm_in_min, min(pcm_in_max, pcm_thickness_inches))  # Clamp to valid range
        pcm_thickness_m = pcm_thickness_inches * IN_TO_M  # Convert to meters
        
        # Update radii with adjustable PCM thickness
        r_pcm = r_st + pcm_thickness_m
        r_ins = r_pcm + self.insulation_thickness_m
        r_mid_pcm = r_st + 0.5 * pcm_thickness_m
        r_mid_ins = r_pcm + 0.5 * self.insulation_thickness_m
        
        # Recalculate vertical areas with new PCM thickness
        A_vert_pcm = 2*math.pi*r_mid_pcm * pcm_thickness_m
        A_vert_ins = 2*math.pi*r_mid_ins * self.insulation_thickness_m
        
        # PCM segmentation parameters
        pcm_segment_thickness_inches = self.pcm_segment_thickness_inches
        pcm_segment_thickness_m = pcm_segment_thickness_inches * IN_TO_M  # inches to meters
        
        def compute_ua_bypass_shell(rc: dict) -> dict:
            """
            Compute UA contributions using only resistors ending in '_AMB'.
            - Bypass UA: keys that start with 'R_WH' and end with '_AMB'
            - Shell UA : all other keys that end with '_AMB'
            Returns a dict with UA_bypass, UA_shell, UA_total.
            """
            ua_bypass = 0.0
            ua_shell  = 0.0

            for key, value in rc.items():
                # Must be a resistance key pointing to ambient
                if not (isinstance(key, str) and key.startswith("R_") and key.endswith("_AMB")):
                    continue

                try:
                    R = float(value)
                except Exception:
                    continue
                if R <= 0.0:
                    continue

                G = 1.0 / R

                # Identify bypass vs shell without regex
                # R_WH{i}_AMB → bypass
                middle = key[2:-4]  # strip 'R_' prefix and '_AMB' suffix
                if middle.startswith("WH") and middle[2:].isdigit():
                    ua_bypass += G
                else:
                    ua_shell += G

            return {
                "UA_bypass": ua_bypass,
                "UA_shell": ua_shell,
                "UA_total": ua_bypass + ua_shell,
            }
        # Robust segmentation algorithm
        def calculate_segments(total_dimension, target_segment_size, min_segments=1, max_segments=1000):
            """Calculate optimal number of segments with robust bounds checking"""
            if total_dimension <= 0:
                return 1, total_dimension
            
            # Calculate ideal number of segments
            n_segments_ideal = total_dimension / target_segment_size
            
            # Round to nearest integer, but enforce bounds
            n_segments = max(min_segments, min(max_segments, round(n_segments_ideal)))
            
            # If we have very thin layers, ensure at least one segment
            if n_segments < 1:
                n_segments = 1
            
            # Calculate actual segment size
            actual_segment_size = total_dimension / n_segments
            
            return n_segments, actual_segment_size
        
        # Calculate radial segments
        n_pcm_radial, actual_pcm_segment_thickness = calculate_segments(
            pcm_thickness_m, pcm_segment_thickness_m, min_segments=1, max_segments=200
        )
        
        # Calculate circumferential segments
        mid_circumference = 2 * math.pi * r_mid_pcm
        n_pcm_circumferential, actual_circumferential_size = calculate_segments(
            mid_circumference, pcm_segment_thickness_m, min_segments=1, max_segments=1
        )
        
        # Calculate axial segments per layer
        n_pcm_axial, actual_pcm_axial_thickness = calculate_segments(
            L_layer, pcm_segment_thickness_m, min_segments=1, max_segments=1
        )
        
        # Validation and warnings
        total_pcm_segments = n_pcm_radial * n_pcm_circumferential * n_pcm_axial * n_nodes
        max_recommended_segments = 50000  # Reasonable limit for computational efficiency
        
        if total_pcm_segments > max_recommended_segments:
            print(f"Warning: Total PCM segments ({total_pcm_segments}) exceeds recommended limit ({max_recommended_segments})")
            print(f"Consider increasing pcm_segment_thickness_inches (currently {pcm_segment_thickness_inches})")
        
        print(f"PCM Configuration:")
        print(f"  - Thickness: {pcm_thickness_inches:.3f} inches ({pcm_thickness_m*1000:.1f} mm)")
        print(f"  - Radial segments: {n_pcm_radial} (thickness: {actual_pcm_segment_thickness*1000:.2f} mm)")
        print(f"  - Circumferential segments: {n_pcm_circumferential} (arc: {actual_circumferential_size*1000:.2f} mm)")
        print(f"  - Axial segments per layer: {n_pcm_axial} (height: {actual_pcm_axial_thickness*1000:.2f} mm)")
        print(f"  - Total PCM segments: {total_pcm_segments}")
        

        # helper: radial conduction
        def R_cond(k, r1, r2):
            return math.log(r2/r1) / (2*math.pi * k * L_layer)

        # compute capacitances & masses per layer
        # enamel
        vol_en = math.pi * (r_en**2-r_w**2) * L_layer
        mass_en = rho_en * vol_en
        C_en = mass_en * c_en
        # steel
        vol_st = math.pi * (r_st**2-r_en**2) * L_layer
        mass_st = rho_st * vol_st
        C_st = mass_st * c_st
        # insulation (updated with new PCM thickness)
        vol_ins = math.pi * (r_ins**2-r_pcm**2) * L_layer
        mass_ins = rho_ins * vol_ins
        C_ins = mass_ins * c_ins

        # PCM segment calculations (updated with adjustable thickness)
        pcm_density = self.pcm_properties["solid"]["pcm_density"] * 1e3 # convert to kg/m³
        cp_pcm = np.interp(start_T,
                           self.enthalpy_lut[:,0],
                           self.enthalpy_lut[:,1])

        # initialize PCM mass dictionary
        self.pcm_mass_dict = {}
        self.pcm_mass_kg = 0
        
    
        # self.water_side_film_h = self.calculate_film_convective_heat_transfer_coefficient()

        self.water_side_film_h = 150
        # build RC network per stratum
        for i in range(1, n_nodes+1):
            # water -> enamel convective
            rc[f"R_WH{i}_ENM{i}"] = 1.0 / (self.water_side_film_h * A_layer)
            # enamel
            rc[f"C_ENM{i}"]      = C_en
            rc[f"R_ENM{i}_STL{i}"] = R_cond(k_en, r_w, r_en)
            # steel
            rc[f"C_STL{i}"]       = C_st
            # Steel connects to innermost PCM segments
            
            # Create PCM segments for this layer
            for r in range(1, n_pcm_radial+1):
                r_inner = r_st + (r-1) * actual_pcm_segment_thickness
                r_outer = r_st + r * actual_pcm_segment_thickness
                r_mid = (r_inner + r_outer) / 2
                
                for c in range(1, n_pcm_circumferential+1):
                    for a in range(1, n_pcm_axial+1):
                        # Segment identifier
                        seg_id = f"PCM{i}-{r}-{c}-{a}"
                        
                        # Calculate segment volume and mass (robust for all PCM thicknesses)
                        theta_segment = 2 * math.pi / n_pcm_circumferential
                        
                        # Volume calculation for cylindrical segment
                        radial_area = math.pi * (r_outer**2 - r_inner**2)
                        circumferential_fraction = theta_segment / (2 * math.pi)
                        vol_segment = radial_area * actual_pcm_axial_thickness * circumferential_fraction
                        
                        # Ensure minimum volume for very thin PCM layers
                        min_vol = 1e-12  # 1 mm³ minimum
                        vol_segment = max(vol_segment, min_vol)
                        
                        mass_segment = pcm_density * vol_segment  # kg
                        
                        # Capacitance
                        rc[f"C_{seg_id}"] = mass_segment * cp_pcm * 1e3 # cp_pcm in in J/g-C
                        
                        if seg_id not in self.pcm_mass_dict:
                            self.pcm_mass_dict[seg_id] = mass_segment
                            self.pcm_mass_kg += mass_segment
                        
                        # Robust resistance calculations with bounds checking
                        def safe_resistance(dr, area, k, min_r=1e-6, max_r=1e6):
                            """Calculate resistance with bounds checking"""
                            if area <= 0 or k <= 0:
                                return max_r
                            r = dr / (k * area)
                            return max(min_r, min(max_r, r))
                        
                        # Radial conduction resistances
                        dr = actual_pcm_segment_thickness
                        A_radial = 2 * math.pi * r_mid * actual_pcm_axial_thickness * circumferential_fraction
                        R_radial = safe_resistance(dr, A_radial, k_pcm)
                        
                        # Connect to steel (innermost radial layer)
                        if r == 1:
                            # Connect to steel
                            rc[f"R_STL{i}_{seg_id}"] = R_radial / 2  # Half resistance to segment center
                        else:
                            # Connect to inner radial neighbor
                            inner_seg_id = f"PCM{i}-{r-1}-{c}-{a}"
                            rc[f"R_{inner_seg_id}_{seg_id}"] = R_radial
                        
                        # Connect to outer radial neighbor or insulation
                        if r == n_pcm_radial:
                            # Connect to insulation
                            rc[f"R_{seg_id}_INS{i}"] = R_radial / 2  # Half resistance to segment center
                        else:
                            # Will be connected when outer segment is created
                            pass
                        
                        # Circumferential conduction resistances
                        dtheta = theta_segment
                        A_circumferential = dr * actual_pcm_axial_thickness
                        R_circumferential = safe_resistance(r_mid * dtheta, A_circumferential, k_pcm)
                        
                        # Connect to circumferential neighbors (wrap around)
                        c_next = c + 1 if c < n_pcm_circumferential else 1
                        c_prev = c - 1 if c > 1 else n_pcm_circumferential
                        
                        next_seg_id = f"PCM{i}-{r}-{c_next}-{a}"
                        prev_seg_id = f"PCM{i}-{r}-{c_prev}-{a}"
                        
                        # Only create resistance to next segment to avoid duplicates
                        if c < c_next or (c == n_pcm_circumferential and c_next == 1 and seg_id != next_seg_id):
                            rc[f"R_{seg_id}_{next_seg_id}"] = R_circumferential
                        
                        # Axial conduction resistances
                        dz = actual_pcm_axial_thickness
                        A_axial = radial_area * circumferential_fraction
                        R_axial_segment = safe_resistance(dz, A_axial, k_pcm)
                        
                        # Connect to axial neighbors within the same layer
                        if a < n_pcm_axial:
                            next_axial_seg_id = f"PCM{i}-{r}-{c}-{a+1}"
                            rc[f"R_{seg_id}_{next_axial_seg_id}"] = R_axial_segment
                        
                        # Connect to adjacent layers (if axial conduction enabled)
                        if include_axial and i < n_nodes:
                            # Connect to corresponding segment in next layer
                            next_layer_seg_id = f"PCM{i+1}-{r}-{c}-{a}"
                            # Use robust resistance calculation
                            R_layer_axial = safe_resistance(dz, A_axial, k_pcm)
                            rc[f"R_{seg_id}_{next_layer_seg_id}"] = R_layer_axial

            # insulation
            rc[f"C_INS{i}"]       = C_ins
            R_ins_r = R_cond(k_ins, r_pcm, r_ins)
            A_out = 2*math.pi*r_ins*L_layer + 2*math.pi*r_ins**2/n_nodes
            rc[f"R_INS{i}_AMB"]   = R_ins_r + 1.0/(h_ext * A_out)

            # axial conduction for non-PCM layers
            if include_axial and i < n_nodes:
                ni = i + 1
                rc[f"R_ENM{i}_ENM{ni}"] = L_layer / (k_en  * A_vert_en)
                rc[f"R_STL{i}_STL{ni}"] = L_layer / (k_st  * A_vert_st)
                rc[f"R_INS{i}_INS{ni}"] = L_layer / (k_ins * A_vert_ins)

        # remove original water->amb resistances
        for i in range(1, n_nodes+1):
            rc.pop(f"R_WH{i}_AMB", None)
        self.rc_params = rc
        print(f"Total PCM Mass (kg): {self.pcm_mass_kg:.2f}")
        
        # Target: make the WH→AMB bypass account for exactly 1/3 of a 2.17 W/K tank UA.
        # Do this by distributing a TARGET **conductance** (not resistance) across nodes,
        # with edge nodes having 2× the conductance of middle nodes (=> edge R = 0.5× middle R).

        target_total_UA = kwargs.get('UA (W/K)', 2.639)                       # [W/K] overall tank UA
        target_bypass_UA = target_total_UA / 3.0     # [W/K] bypass should be one-third of total

        # Conductance weights per node: edges = 2, middles = 1 (you can change edge_weight if desired)
        edge_weight = 2.0
        weights = [edge_weight] + [1.0]*(n_nodes-2) + [edge_weight] if n_nodes > 1 else [1.0]
        w_sum = float(sum(weights))

        # Distribute the target bypass conductance over nodes:
        #   G_i = (w_i / sum(w)) * target_bypass_UA
        #   R_i = 1 / G_i
        # This ensures: sum_i (1/R_i) == target_bypass_UA and R_edge = 0.5 * R_middle.
        for i in range(1, n_nodes + 1):
            G_i = (weights[i - 1] / w_sum) * target_bypass_UA
            R_i = 1.0 / max(G_i, 1e-12)
            rc[f"R_WH{i}_AMB"] = R_i
            
        tank_ua_values = compute_ua_bypass_shell(rc)
        self.tank_ua_values = tank_ua_values
        print(tank_ua_values)
        return rc
    
    def update_rc_network(self, t_pcm, states, current_schedule, **kwargs):
        '''Get the dynamic specific heat for each of the pcm nodes and update the capacitance in the rc_network'''
        
        pcm_specific_heats = np.interp(t_pcm, self.key_temp, self.key_specific_heats)

        # Loop over each PCM node and apply the modifications
        self.water_side_film_h = self.calculate_film_convective_heat_transfer_coefficient(states, current_schedule)
        A_layer = self.internal_area_m2 / self.n_nodes
        film_conv_resistance = 1/(self.water_side_film_h * A_layer)
        
        # Update in-place; collect whether each expected key existed
        for i in range(1, self.n_nodes + 1):
            key = f"R_WH{i}_ENM{i}"
            if key in self.rc_params:
                self.rc_params[key] = float(film_conv_resistance[i-1])


        for i, node in enumerate(self.pcm_mass_dict.items()):
            
            index = node[0]
            pcm_node_mass = node[1]
            pcm_specific_heat = pcm_specific_heats[i]

            
            # Update the capacitance for this node (in J/K)
            self.rc_params[f"C_{index}"] = pcm_specific_heat * pcm_node_mass * 1e3
            # self.rc_params[f"C_PCM{node}"] = 1e-3
        return self.rc_params
        
    def update_state_space_model(self, states, current_schedule, **kwargs):
        
        pcm_temps = self.next_states[self.t_pcm_idx]
        dynamic_rc_params = self.update_rc_network(pcm_temps, states, current_schedule)
        all_cap = {name.upper().split('_')[1:][0]: val 
               for name, val in dynamic_rc_params.items() if name[0] == 'C'}
        all_res = {tuple(name.upper().split('_')[1:]): val 
                for name, val in dynamic_rc_params.items() if name[0] == 'R'}
        
        # You may need to re-identify internal/external nodes if not stored already
        internal_nodes = [node for node in all_cap.keys()]
        
        external_nodes = [node for node in self.external_nodes]  # assuming these were stored
        
        # Recompute the state-space matrices
        A_c, B_c = self.create_rc_matrices(all_cap, all_res, internal_nodes, external_nodes)
        
        # Update the model’s matrices
        self.A = A_c
        self.B = B_c
        self.capacitances = np.array(list(all_cap.values()))
        
        # Define state and input names
        state_names = ['T_' + node for node in internal_nodes]
        input_names = ['T_' + node for node in external_nodes] + ['H_' + node for node in internal_nodes]
        
        outputs = None
        matrices=(A_c, B_c)
        
        # if unused_inputs is not None:
        #     good_input_idx = [i for (i, name) in enumerate(input_names) if name not in unused_inputs]
        #     B_c = B_c[:, good_input_idx]
        #     input_names = [name for name in input_names if name not in unused_inputs]

        # initialize states based on matrices
        
        # Define states
        self.nx = len(self.states)
        # if isinstance(self.states, dict):
        #     self.states = np.array(list(self.states.values()), dtype=float)
        # else:
        #     self.states = np.zeros(self.nx, dtype=float)

        # Define inputs
        self.nu = len(input_names)
        # if isinstance(input_names, dict):
        #     self.inputs = np.array(list(input_names.values()), dtype=float)
        # else:
        #     self.inputs = np.zeros(self.nu, dtype=float)
        self.input_names = list(input_names)
        self.use_schedule_for_inputs = all([col in self.input_names for col in self.schedule.columns])
        self.inputs_init = self.inputs  # for saving values from update_inputs step
        
        # Define outputs
        if outputs is None:
            self.ny = self.nx
            self.output_names = self.state_names.copy()
        else:
            self.ny = len(outputs)
            self.output_names = outputs
        # self.outputs = np.zeros(self.ny, dtype=float)
        # self.next_outputs = self.outputs  # for saving outputs of next time step

        # Define continuous-time matrices
        self.A_c, self.B_c, self.C, self.D = self.create_matrices(matrices)

        # Update output values
        self.outputs = self.C.dot(self.states) + self.D.dot(self.inputs)

        # Reduce model order (i.e. number of states)
        self.reduced = False
        self.transformation_matrix = None
        if 'reduced_states' in kwargs or 'reduced_min_accuracy' in kwargs:
            self.reduce_model(update_discrete=False, **kwargs)

        # Create A, B discrete matrices
        self.A, self.B = self.to_discrete()


    def get_pcm_heat_xfer(self):
        # if convection coefficient changes by phase, add heat transfer here


        return np.array([0] * len(self.t_pcm_idx))

        # # calculate heat transfer (pcm to water)
        # t_water = self.states[self.t_pcm_wh_idx]
        # t_pcm = self.states[self.t_pcm_idx]
        # h_pcm = PCM_PROPERTIES["h_conv"] * (t_pcm - t_water)  # in W
        # return h_pcml
        return 0 # keep zero for now to prevent double counting pcm heat transfer

    
    def calculate_film_convective_heat_transfer_coefficient(self, states, current_schedule):
        """
        Water-side film heat transfer coefficient h_conv [W/m^2-K], **per node**, with mixed convection.

        Per-node behavior:
        • Uses node-aligned T_fluid and T_wall.
        • Natural-convection Nu is computed **per node**.
        • Forced-convection Nu may vary **per node** via self.u_wall_factor_by_node (optional, shape = n_nodes).
            - If absent, uses ones (same near-wall sweep at each node).
        • Mixed blend is done **per node**.

        Draw rate source:
        • self.draw_total (preferred). Units handling:
            - If self.draw_total_units in {'L/min','l/min','lpm','l per min','l_per_min'} → convert to m^3/s.
            - Else if value > 1.0 → assume L/min; otherwise assume m^3/s.

        Stability rule for natural convection at the wall:
        • If wall hotter than fluid locally (stable), set Nu_nat = 1.0 (conduction limit).

        Returns:
        h_z : np.ndarray (n_nodes,) — node-wise film coefficients.
        """
        import numpy as np

        # ---------------- helpers ----------------
        def _ensure_C(arr):
            arr = np.asarray(arr, dtype=float)
            return arr - 273.15 if arr.size and np.nanmedian(arr) > 200.0 else arr

        def _to_K(arr_C):
            return np.asarray(arr_C, dtype=float) + 273.15

        def water_props_C(Tc):
            """(rho [kg/m3], mu [Pa·s], k [W/m·K], cp [J/kg·K], Pr [-]); valid ~20–80°C."""
            Tk = Tc + 273.15
            rho = float(np.clip(1000.0 - 0.3*(Tc - 20.0), 950.0, 1000.0))
            mu  = 2.414e-5 * 10.0**(247.8/(Tk - 140.0))             # Pa·s (Andrade)
            k   = float(np.clip(0.561 + 0.0018*(Tc - 20.0), 0.55, 0.68))
            cp  = 4180.0
            Pr  = cp * mu / max(k, 1e-12)
            return rho, mu, k, cp, Pr

        def water_density_and_beta_lut(T_C):
            """
            LUT-based density and beta for water at 1 atm.

            Inputs:
              T_C : np.ndarray [°C]

            Outputs:
              rho  : density [kg/m³]
              beta : volumetric thermal expansion [1/°C],
                     beta = -(1/rho) * dρ/dT from LUT slope
            """
            _TEMPS = np.array([0, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
            _DENSITIES = np.array(
                [
                    999.8,
                    1000.0,
                    999.7,
                    998.2,
                    995.7,
                    992.2,
                    988.1,
                    983.2,
                    977.8,
                    971.8,
                    965.3,
                    958.4,
                ],
                dtype=float,
            )

            T = np.asarray(T_C, dtype=float)
            T_clipped = np.clip(T, _TEMPS[0], _TEMPS[-1])

            # Interpolated density
            rho = np.interp(T_clipped, _TEMPS, _DENSITIES)

            # Piecewise-linear slope dρ/dT between LUT knots
            idx = np.searchsorted(_TEMPS, T_clipped, side="right") - 1
            idx = np.clip(idx, 0, len(_TEMPS) - 2)

            d_rho = _DENSITIES[idx + 1] - _DENSITIES[idx]
            d_T   = _TEMPS[idx + 1] - _TEMPS[idx]
            drho_dT = d_rho / d_T

            rho_safe = np.where(rho == 0.0, 1e-12, rho)
            beta = - (drho_dT / rho_safe)

            return rho, beta

        # ---------- geometry ----------
        r = float(self.tank_radius_m)
        L = float(getattr(self, "tank_height_m", 2.0 * r))
        
        # Characteristic length of each node
        L_node = L / self.n_nodes
        if L <= 0.0 or r <= 0.0:
            raise ValueError("tank_height_m and tank_radius_m must be > 0")
        A_cs = np.pi * r * r

        # Near-wall sweep baseline for mixed convection (vertical wall)
        gamma_w = float(getattr(self, "mixed_conv_wall_sweep_factor", 0.3))
        gamma_w = float(np.clip(gamma_w, 0.05, 1.5))

        # ---------- node temperatures ----------
        T_fluid_C = _ensure_C(self.states[self.t_wh_idx])
        T_wall_C  = _ensure_C(self.states[self.t_en_idx]) if getattr(self, "t_en_idx", None) is not None \
                    else np.full_like(T_fluid_C, float(np.nanmean(T_fluid_C)) if T_fluid_C.size else 50.0)

        n_nodes = min(T_fluid_C.size, T_wall_C.size)
        if n_nodes == 0:
            h_default = float(getattr(self, "water_side_film_h", 10.0))
            return np.array([h_default], dtype=float)

        # Trim & convert
        T_fluid_C = T_fluid_C[:n_nodes]
        T_wall_C  = T_wall_C[:n_nodes]
        T_fluid_K = _to_K(T_fluid_C)
        T_wall_K  = _to_K(T_wall_C)
        dT = T_wall_K - T_fluid_K  # sign used only for heat-flux direction outside; |dT| for Nu

        # ---------- properties at mean fluid T ----------
        T_mean_C = float(np.nanmean(T_fluid_C))
        rho_mean, mu, k_f, cp, Pr_f = water_props_C(T_mean_C)
        nu_f = mu / max(rho_mean, 1e-12)
        g = 9.81

        # water beta_z using LUT-based density slope
        rho_nodes, beta_z = water_density_and_beta_lut(T_fluid_C)  # beta_z is per node

        # ---------- draw rate → velocity ----------
        Q_draw_val = float(max(getattr(self, "draw_total", 0.0), 0.0))
        units = getattr(self, "draw_total_units", None)
        if isinstance(units, str) and units.lower() in ("l/min", "lpm", "l per min", "l_per_min"):
            Q_draw_m3s = Q_draw_val / 1000.0 / 60.0
        else:
            Q_draw_m3s = Q_draw_val / 1000.0 / 60.0 if Q_draw_val > 1.0 else Q_draw_val
        U_mean = Q_draw_m3s / max(A_cs, 1e-12)

        # Optional per-node factor for near-wall sweep
        u_scale = getattr(self, "u_wall_factor_by_node", None)
        if u_scale is None:
            u_scale = np.ones(n_nodes, dtype=float)
        else:
            u_scale = np.asarray(u_scale, dtype=float)
            if u_scale.size != n_nodes:
                raise ValueError("u_wall_factor_by_node must have length n_nodes")
            u_scale = np.clip(u_scale, 0.0, 5.0)

        U_wall_node = gamma_w * U_mean * u_scale
        Re_L_node = U_wall_node * L_node / max(nu_f, 1e-12)

        # top and bottom walls ignored
        # ---------- natural convection (vertical wall) PER NODE ----------
        C_lam = 0.671 / (1.0 + (0.492 / max(Pr_f, 1e-12))**0.5625)**0.444
        Ra_side = g * beta_z * np.abs(dT) * (L_node**3) / (max(nu_f, 1e-12)**2) * Pr_f
        Ra_quarter = np.maximum(Ra_side, 1e-30)**0.25
        Nu_nat = 2.0 / np.log(np.maximum(1.0 + 2.0 / np.maximum(C_lam * Ra_quarter, 1.0e-30), 1.0 + 1.0e-12))
        Nu_nat = np.maximum(Nu_nat, 1.0)  # conduction floor

        # ---------- forced convection (vertical plate) PER NODE ----------
        Pr13 = Pr_f**(1.0/3.0)
        Nu_forced = np.where(
            Re_L_node < 5.0e5,
            0.664 * np.sqrt(np.maximum(Re_L_node, 0.0)) * Pr13,
            np.maximum(0.037 * (np.maximum(Re_L_node, 0.0)**0.8) * Pr13 - 871.0 * Pr13, 0.0),
        )
        if Q_draw_m3s <= 0.0:
            Nu_forced[:] = 0.0

        # ---------- mixed convection PER NODE ----------
        n_blend = 3.0
        Nu_tot = (Nu_forced**n_blend + Nu_nat**n_blend)**(1.0/n_blend)

        # ---------- h per node ----------
        h_z = Nu_tot * k_f / L_node

        # if hasattr(self, "water_side_film_h") and self.water_side_film_h is not None:
        #     href = float(self.water_side_film_h)
        #     h_z = np.clip(h_z, 0.1 * href, 10.0 * href)
        # else:
        #     h_z = np.clip(h_z, 5e-3, 5e3)

        h_z = np.clip(h_z, 1e-6, 1e6)

        return h_z


    def update_inputs(self, schedule_inputs=None):
        # Note: self.inputs_init are not updated here, only self.current_schedule
        super().update_inputs(schedule_inputs)
        length = self.nu - len(self.inputs_init)
        
        zeros_array = np.zeros(length)
        self.inputs_init = np.append(self.inputs_init, zeros_array)
        

        # get heat injections from PCM
        # self.pcm_heat_to_water = self.get_pcm_heat_xfer()

        # # # add PCM heat to inputs       
        # self.inputs_init = np.append(self.inputs_init, -self.pcm_heat_to_water) # fix heat flow direction
        # # # self.inputs_init[self.h_pcm_idx] = self.pcm_heat_to_water
        # self.inputs_init[1:13] += self.pcm_heat_to_water # fix heat flow direction

    # def update_model(self, control_signal=None):
    #     super().update_model(control_signal)

    #     # calculate new PCM enthalpy based on linear temperature change
    #     delta_t = self.next_states[self.t_pcm_idx] - self.states[self.t_pcm_idx] 
    #     q_pcm = delta_t * self.capacitances[self.t_pcm_idx] # J
    #     self.pcm_heat_to_water_rc_network = -q_pcm/self.time_res.total_seconds()
    #     self.enthalpy_pcm += q_pcm

    #     # update PCM temperature in new_states
    #     t_pcm = np.interp(self.enthalpy_pcm, self.key_enthalpy, self.key_temp)
    #     self.next_states[self.t_pcm_idx] = t_pcm
    #     self.update_state_space_model()
        
    def update_model(self, control_signal=None, epsilon=None, max_iter=None, iter_count=0,
                    original_states=None, original_inputs=None, original_inputs_init=None):
        # Use provided convergence criteria or fall back to instance attributes.
        # if epsilon is None:
        #     epsilon = self.epsilon
        # if max_iter is None:
        #     max_iter = self.max_iter

        # # On the first call, store copies of the original states, inputs, and inputs_init.
        # if original_states is None:
        #     original_states = self.states.copy()
        # if original_inputs is None:
        #     original_inputs = self.inputs.copy()
        # if original_inputs_init is None:
        #     original_inputs_init = self.inputs_init.copy()

        # Call the parent's update_model function.
        super().update_model(control_signal)
        
        # Only use temperatures output from state space model and enthalpies nothing else
        # multiply those values out by the mass of pcm
        masses = np.fromiter(self.pcm_mass_dict.values(), dtype=float) * 1e3 
        enthalpy_state = np.interp(self.states[self.t_pcm_idx], self.key_temp, self.key_enthalpy) * masses
        enthalpy_next_state = np.interp(self.next_states[self.t_pcm_idx], self.key_temp, self.key_enthalpy) * masses
        q_pcm = enthalpy_next_state - enthalpy_state
        self.pcm_heat_to_water_rc_network = -q_pcm / self.time_res.total_seconds()
        self.enthalpy_pcm = enthalpy_next_state
        self.delta_enthalpy_pcm = enthalpy_next_state - enthalpy_state
        

        # Update the PCM temperature using interpolation from enthalpy to temperature.
        # t_pcm = np.interp(self.enthalpy_pcm, self.key_enthalpy, self.key_temp)
        # t_pcm = self.next_states[self.t_pcm_idx]
        # self.next_states[self.t_pcm_idx] = t_pcm

        # Call the state space model update (which may modify states, inputs, and inputs_init).
        self.update_state_space_model(self.states, self.current_schedule)

        # Reset the states and inputs to their original values so they stay constant between iterations.
        # self.states = original_states.copy()
        # self.inputs = original_inputs.copy()
        # self.inputs_init = original_inputs_init.copy()

        # # Check convergence: if the maximum change is smaller than epsilon or max iterations reached, stop.
        # if np.max(np.abs(delta_t)) < epsilon or iter_count > max_iter:
        #     self.step_num += 1
        #     print(f"Convergence reached after {iter_count + 1} iterations: ΔT = {np.max(np.abs(delta_t)):.6f} [{self.step_num}/{self.sim_times.size}]")
        #     return
        # else:
        #     # Update the current state for the next iteration.
        #     self.states[self.t_pcm_idx] = t_pcm

        #     # Recursively call update_model with the original copies maintained.
        #     self.update_model(control_signal, epsilon, max_iter, iter_count + 1,
        #                     original_states, original_inputs, original_inputs_init)
        
    def generate_results(self):
        # Note: most results are included in Dwelling/WH. Only inputs and states are saved to self.results
        results = super().generate_results()

        if self.verbosity >= 3:
            results['Total Water Heater PCM Enthalpy (J)'] = self.enthalpy_pcm.sum()
            results['Delta Water Heater PCM Enthalpy (J)'] = self.delta_enthalpy_pcm.sum()
            results['Total Water Heater PCM Heat Injected (W)'] = self.pcm_heat_to_water_rc_network.sum()
            
        if self.verbosity >= 6:
            # water nodes
            for i, idx in enumerate(self.t_wh_idx):
                results[f'Film Tank {self.state_names[idx]} Heat Transfer Coefficient (W/m^2-K)'] = self.water_side_film_h[i]            
            # pcm nodes
            for i, idx in enumerate(self.t_pcm_idx):
                results[f"Water Tank {self.state_names[idx]} Temperature (C)"] = self.states[idx]
                results[f"Water Tank {self.state_names[idx]} Water Temperature (C)"] = self.states[idx]
                results[f"Water Tank {self.state_names[idx]} Enthalpy (J)"] = self.enthalpy_pcm[i]
                results[f"Water Tank {self.state_names[idx]} Heat Injected (W)"] = self.pcm_heat_to_water_rc_network[i]
                results[f"Water Tank {self.state_names[idx]} Capacitance (J/K)"] = self.capacitances[idx]
                
            results['PCM Mass (kg)'] = self.pcm_mass_kg
            for key,value in self.tank_ua_values.items():
                results[key] = value
            results['Water Volume (L)'] = self.volume

            
        return results
    
    
    # TODO FILM heat transfer coefficient changes with the water tank flow rate





