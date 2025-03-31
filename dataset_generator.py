# SPDX-License-Identifier: MIT
import time
import argparse
import pandas as pd
import pandapower as pp
import numpy as np
import networkx as nx
import multiprocessing as mp
import os
import simbench as sb

from utils.data_utils import perturb_topology

number_of_samples = 1000
number_of_processes = 126
randomize_loads = True

case = "ehub"
grid_codes=['1-MV-urban--2-sw','1-MV-rural--2-no_sw','1-MV-comm--2-no_sw']
max_pu_deviation = 0.04
target_line_load = 60

der_configurations = [{'grid_code': '1-MV-urban--2-sw', 'der_node': 80, 'cong_line': 74},
                {'grid_code': '1-MV-comm--2-no_sw', 'der_node': 80, 'cong_line': 74},
                {'grid_code': '1-MV-rural--2-no_sw', 'der_node': 49, 'cong_line': 45},
                {'grid_code': '1-MV-semiurb--2-no_sw', 'der_node': 31, 'cong_line': 27},
                {'grid_code': '1-HVMV-mixed-all-2-no_sw', 'der_node': 42, 'cong_line': 79}]

ENFORCE_Q_LIMS = False

def create_case3():
    net = pp.create_empty_network()
    net.sn_mva = 100
    b0 = pp.create_bus(net, vn_kv=345., name='bus 0')
    b1 = pp.create_bus(net, vn_kv=345., name='bus 1')
    b2 = pp.create_bus(net, vn_kv=345., name='bus 2')
    pp.create_ext_grid(net, bus=b0, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=b2, p_mw=10.3, q_mvar=3, name="Load")
    pp.create_line(net, from_bus=b0, to_bus=b1, length_km=10, name='line 01', std_type='NAYY 4x50 SE')
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=5, name='line 01', std_type='NAYY 4x50 SE')
    pp.create_line(net, from_bus=b2, to_bus=b0, length_km=20, name='line 01', std_type='NAYY 4x50 SE')
    
    net.line['c_nf_per_km'] = pd.Series(0., index=net.line['c_nf_per_km'].index, name=net.line['c_nf_per_km'].name)
    
    return net

def remove_c_nf(net):
    net.line['c_nf_per_km'] = pd.Series(0., index=net.line['c_nf_per_km'].index, name=net.line['c_nf_per_km'].name)
    
def unify_vn(net):
    for node_id in range(net.bus['vn_kv'].shape[0]):
        net.bus['vn_kv'][node_id] = max(net.bus['vn_kv'])

def get_trafo_z_pu(net):
        
    net.trafo.loc[net.trafo.index, 'i0_percent'] = 0.
    net.trafo.loc[net.trafo.index, 'pfe_kw'] = 0.
    
    z_pu = net.trafo['vk_percent'].values / 100. * 1000. / net.sn_mva
    r_pu = net.trafo['vkr_percent'].values / 100. * 1000. / net.sn_mva
    x_pu = np.sqrt(z_pu**2 - r_pu**2)
    
    return x_pu, r_pu
    
def get_line_z_pu(net):
    r = net.line['r_ohm_per_km'].values * net.line['length_km'].values
    x = net.line['x_ohm_per_km'].values * net.line['length_km'].values
    from_bus = net.line['from_bus']
    to_bus = net.line['to_bus']
    vn_kv_to = net.bus['vn_kv'][to_bus].to_numpy()
    zn = vn_kv_to**2 / net.sn_mva
    r_pu = r/zn
    x_pu = x/zn
    
    return r_pu, x_pu

def get_adjacency_matrix(net):
    multi_graph = pp.topology.create_nxgraph(net)
    A = nx.adjacency_matrix(multi_graph).todense() 
    
    return A

def get_optimal_der_power(working_net, der_node, cong_line, max_load_per_node=5, print_debug=True):
    '''Reduces the load at the der_node to a certain level, while keeping the line load below a certain level'''
    
    load_index = working_net.load[working_net.load["bus"] == der_node].index[0]
    
    starting_load = working_net.load['p_mw'][load_index]
    if abs(working_net.load['p_mw'][load_index]) >= max_load_per_node:
        working_net.load['p_mw'][load_index] = 0

    load_adjustment_summand = max_load_per_node / 5
    
    pp.runpp(working_net, numba=False)

    cong_line_load = working_net.res_line.loading_percent[cong_line]
    
    if print_debug:
        print(f"ehub line load {cong_line_load}")
        
    max_iterations = 100
    iteration = 0
    while abs(load_adjustment_summand) > 0.01 and abs(working_net.load['p_mw'][load_index]) < max_load_per_node and iteration < max_iterations:
        working_net.load.loc[load_index, 'p_mw'] -= load_adjustment_summand

        pp.runpp(working_net, numba=False)

        current_line_load_max = working_net.res_line.loading_percent[cong_line]

        if abs(current_line_load_max - target_line_load) > abs(cong_line_load - target_line_load):
            if print_debug:
                print('wrong direction')
            load_adjustment_summand *= -0.5

        cong_line_load = current_line_load_max
        iteration += 1

    final_load = min(max(working_net.load['p_mw'][load_index], -max_load_per_node), max_load_per_node)

    if print_debug:
        print(f"Final load: {final_load}")

    return final_load - starting_load



def generate_data(sublist_size, rng, grid_code, der_node, cong_line, randomized_loads, random_fact, num_lines_to_remove=0, num_lines_to_add=0, timestamps=[0]):
    edge_features_list = []
    node_features_list = []

    ts_iterator = 0
    overload_timesteps = []
    while ts_iterator < len(timestamps):
        ts = timestamps[ts_iterator]
        ts_iterator += 1

        net=sb.get_simbench_net(grid_code)
        remove_c_nf(net)
        
        success_flag, net = perturb_topology(net, num_lines_to_remove=num_lines_to_remove, num_lines_to_add=num_lines_to_add) # TODO 
        if success_flag == 1:
            exit()
        n = net.bus.values.shape[0]
        
        net.bus['name'] = net.bus.index

        r = net.line['r_ohm_per_km'].values    
        x = net.line['x_ohm_per_km'].values
        le = net.line['length_km'].values

        Pg = net.gen['p_mw'].values
        Pd = net.load['p_mw'].values
        Qd = net.load['q_mvar'].values
        r = rng.uniform(0.8*r, 1.2*r, r.shape[0])
        _x_min = np.where(x>=0, 0.8*x, 1.2*x)
        _x_max = np.where(x>=0, 1.2*x, 0.8*x)
        x = rng.uniform(_x_min, _x_max, x.shape[0])
        le = rng.uniform(0.8*le, 1.2*le, le.shape[0])
        Pg = rng.normal(Pg, 0.1*np.abs(Pg), net.gen['p_mw'].shape[0])
        Pd = rng.normal(Pd, 0.1*np.abs(Pd), net.load['p_mw'].shape[0])
        Qd = rng.normal(Qd, 0.1*np.abs(Qd), net.load['q_mvar'].shape[0])
        # apply simbench profiles to the network of timestep ts
        profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
        for elm_param in profiles.keys():
            if profiles[elm_param].shape[1]:
                elm = elm_param[0]
                param = elm_param[1]
                P_or_Q = profiles[elm_param].loc[ts]
                if randomized_loads:
                    P_or_Q = rng.normal(P_or_Q, random_fact*np.abs(P_or_Q))
                net[elm].loc[:, param] = P_or_Q

        # Reset the bus index to make it continuous
        old_index = net.bus.index.copy()
        net.bus.reset_index(drop=True, inplace=True)

        # Create a dictionary to map the old bus indices to the new indices
        bus_index_mapping = {old: new for old, new in zip(old_index, net.bus.index)}

        # Update the from_bus and to_bus columns in the line DataFrame
        net.line['from_bus'] = net.line['from_bus'].map(bus_index_mapping)
        net.line['to_bus'] = net.line['to_bus'].map(bus_index_mapping)
        net.trafo['hv_bus'] = net.trafo['hv_bus'].map(bus_index_mapping)
        net.trafo['lv_bus'] = net.trafo['lv_bus'].map(bus_index_mapping)
        net.sgen['bus'] = net.sgen['bus'].map(bus_index_mapping)
        net.load['bus'] = net.load['bus'].map(bus_index_mapping)
        net.switch['bus'] = net.switch['bus'].map(bus_index_mapping)

        net.storage['bus'] = net.storage['bus'].map(bus_index_mapping)
        if len(net.load[net.load["bus"] == bus_index_mapping[der_node]]) == 0:
            try:
                pp.create_load(net, bus=bus_index_mapping[der_node], p_mw=0, q_mvar=0, name="EHUB")
            except:
                print("komische pp exception")
                import pandapower as pp
                pp.create_load(net, bus=bus_index_mapping[der_node], p_mw=0, q_mvar=0, name="EHUB")
        
        try:
            net['converged'] = False
            pp.runpp(net, algorithm='nr', init="results", numba=False, enforce_q_lims=ENFORCE_Q_LIMS)
        except:
            if not net['converged']:
                print(f'Failed to converge, current sample number: {len(edge_features_list)}')
                import pandapower as pp # dont know why but this helps
                continue
        edge_features = np.zeros((net.line.shape[0], 5))
        edge_features[:, 0] = net.line['from_bus'].values
        edge_features[:, 1] = net.line['to_bus'].values
        edge_features[:, 2], edge_features[:, 3] = get_line_z_pu(net)
        
        
        trafo_edge_features = np.zeros((net.trafo.shape[0], 5))
        trafo_edge_features[:, 0] = net.trafo['hv_bus'].values
        trafo_edge_features[:, 1] = net.trafo['lv_bus'].values
        trafo_edge_features[:, 2], trafo_edge_features[:, 3] = get_trafo_z_pu(net)
        
        edge_features = np.concatenate((edge_features, trafo_edge_features), axis=0)
        types = np.ones(n)*2 # type = load
        for j in range(net.gen.shape[0]):
            index = np.where(net.gen['bus'].values[j] == net.bus['name'])[0][0] 
            if ENFORCE_Q_LIMS:
                if net.res_gen['q_mvar'][j] <= net.gen['min_q_mvar'][j] + 1e-6 \
                    or net.res_gen['q_mvar'][j] >= net.gen['max_q_mvar'][j] - 1e-6:
                        continue # seen as load bus
            types[index] = 1  # type = generator
        for j in range(net.ext_grid.shape[0]):
            index = np.where(net.ext_grid['bus'].values[j] == net.bus['name'])[0][0]
            types[index] = 0 # type = slack bus
        node_features = np.zeros((n, 8))
        node_features[:, 0] = net.bus['name'].values # index
        node_features[:, 1] = types  # type
        node_features[:, 2] = net.res_bus['vm_pu']  # Vm
        node_features[:, 3] = net.res_bus['va_degree']  # Va
        node_features[:, 4] = net.res_bus['p_mw'] / net.sn_mva    # P / pu
        node_features[:, 5] = net.res_bus['q_mvar'] / net.sn_mva  # Q / pu
        opt_der_power = 0

        print(f"current line loading {net.res_line.loading_percent[bus_index_mapping[der_node]]} with target {target_line_load}")
        if net.res_line.loading_percent[cong_line] >= target_line_load:
            opt_der_power = get_optimal_der_power(net, der_node=bus_index_mapping[der_node], cong_line=cong_line, max_load_per_node=5, print_debug=True)

        node_features[:, 6][net.load[net.load["bus"] == bus_index_mapping[der_node]].index[0]] = opt_der_power / net.sn_mva
        node_features[:, 7][bus_index_mapping[der_node]] = 1
        edge_features[:, 4][cong_line] = 1

        edge_features_list.append(edge_features)
        node_features_list.append(node_features)

        if len(edge_features_list) % 10 == 0 or len(edge_features_list) == sublist_size:
            print(f'[Process {os.getpid()}] Current sample number: {len(edge_features_list)}')
        
        
            

    return edge_features_list, node_features_list, overload_timesteps

def generate_data_parallel(num_samples, num_processes, grid_code, der_node, cong_line, randomized_loads, random_fact, num_lines_to_remove=0, num_lines_to_add=0):
    sublist_size = num_samples // num_processes
    timestamps = [list(range(i * sublist_size, (i + 1) * sublist_size)) for i in range(num_processes)]
    parent_rng = np.random.default_rng(234)
    streams = parent_rng.spawn(num_processes)
    pool = mp.Pool(processes=num_processes)
    args = [[sublist_size, streams[0], grid_code, der_node, cong_line, randomized_loads, random_fact, num_lines_to_remove, num_lines_to_add, ts] for ts in timestamps]
    results = pool.starmap(generate_data, args)
    pool.close()
    pool.join()
    
    edge_features_list = []
    node_features_list = []
    overload_timesteps = []
    for sub_res in results:
        edge_features_list += sub_res[0]
        node_features_list += sub_res[1]
        overload_timesteps += sub_res[2]
        
    return edge_features_list, node_features_list, overload_timesteps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Power Flow Data Generator', description='')
    parser.add_argument('--case', type=str, default='simbench_mv_semiurb', help='e.g. 118, 14, 6470rte')
    parser.add_argument('--num_lines_to_remove', '-r', type=int, default=0, help='Number of lines to remove')
    parser.add_argument('--num_lines_to_add', '-a', type=int, default=0, help='Number of lines to add')
    args = parser.parse_args()

    num_lines_to_remove = args.num_lines_to_remove
    num_lines_to_add = args.num_lines_to_add

    if num_lines_to_remove > 0 or num_lines_to_add > 0:
        complete_case_name = 'case' + case + 'perturbed' + f'{num_lines_to_remove:1d}' + 'r' + f'{num_lines_to_add:1d}' + 'a'
    else:
        complete_case_name = 'caseLine' + case
    
    # Generate data
    config_counter = 0
    
    while os.path.exists("./data/raw/"+complete_case_name+str(config_counter)+"_edge_features.npy"):
        config_counter += 1

    config_list = der_configurations

    for ehconfig in der_configurations[4:]:
        edge_features_list, node_features_list, overload_timesteps = generate_data_parallel(number_of_samples, number_of_processes, grid_code = ehconfig['grid_code'],
                                                                                            der_node=ehconfig['der_node'], cong_line=ehconfig['cong_line'], randomized_loads=True, random_fact=0.6, num_lines_to_remove=num_lines_to_remove,
                                                                                            num_lines_to_add=num_lines_to_add)
        
        # Turn the lists into numpy arrays
        edge_features = np.array(edge_features_list)
        node_features = np.array(node_features_list)

        # Print the shapes
        print(f'edge_features shape: {edge_features.shape}')
        print(f'node_features_x shape: {node_features.shape}')

        print(f'range of edge_features "from": {np.min(edge_features[:,:,0])} - {np.max(edge_features[:,:,0])}')
        print(f'range of edge_features "to": {np.min(edge_features[:,:,1])} - {np.max(edge_features[:,:,1])}')
        print(f'range of node_features "index": {np.min(node_features[:,:,0])} - {np.max(node_features[:,:,0])}')


        # delete processed data if exists
        if os.path.exists("./data/processed/"+complete_case_name+str(config_counter)+"_edge_features.npy"):
            os.remove("./data/processed/"+complete_case_name+str(config_counter)+"_edge_features.npy")

        if os.path.exists("./data/processed/"+complete_case_name+str(config_counter)+"_node_features.npy"):
            os.remove("./data/processed/"+complete_case_name+str(config_counter)+"_node_features.npy")
        
        if os.path.exists("./data/processed/"+complete_case_name+str(config_counter)+"DER_edge_features.npy"):
            os.remove("./data/processed/"+complete_case_name+str(config_counter)+"DER_edge_features.npy")
        
        if os.path.exists("./data/processed/"+complete_case_name+str(config_counter)+"DER_node_features.npy"):
            os.remove("./data/processed/"+complete_case_name+str(config_counter)+"DER_node_features.npy")

        # save the features
        os.makedirs("./data/raw", exist_ok=True)
        with open("./data/raw/"+complete_case_name+str(config_counter)+"_edge_features.npy", 'wb') as f:
            np.save(f, edge_features)

        with open("./data/raw/"+complete_case_name+str(config_counter)+"_node_features.npy", 'wb') as f:
            np.save(f, node_features)
        
        config_counter += 1