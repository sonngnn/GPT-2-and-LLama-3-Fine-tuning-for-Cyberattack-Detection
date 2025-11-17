# Feature_extraction.py

import dpkt
import pandas as pd
import numpy as np
from scapy.all import *
from Communication_features import Communication_wifi, Communication_zigbee
from Connectivity_features import Connectivity_features_basic, Connectivity_features_time, \
    Connectivity_features_flags_bytes
from Dynamic_features import Dynamic_features
from Layered_features import L3, L4, L2, L1
from Supporting_functions import get_protocol_name, get_flow_info, get_flag_values, compare_flow_flags, \
    get_src_dst_packets, calculate_incoming_connections, \
    calculate_packets_counts_per_ips_proto, calculate_packets_count_per_ports_proto
    
from tqdm import tqdm
import time
import datetime

category = "DDOS"
attack_type = "SYN_FLOOD"
protocol = "TCP"

class Feature_extraction():
    columns = ["flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
               "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
               "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count",
               "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP",
               "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT",
               "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight", "label", 
               "cumulative_duration", "remaining_time"]    
    
    def pcap_evaluation(self, pcap_file, csv_file_name):
        print(f"[INFO] Traitement du fichier : {pcap_file}")
        
        # Initialisation des variables globales
        ethsize = []
        src_ports, dst_ports = {}, {}
        src_ips, dst_ips, ips = set(), set(), set()
        tcpflows, udpflows = {}, {}
        src_packet_count, dst_packet_count = {}, {}
        src_ip_byte, dst_ip_byte = {}, {}
        protcols_count = {}
        tcp_flow_flags = {}
        incoming_packets_src, incoming_packets_dst = {}, {}
        packets_per_protocol = {}
        average_per_proto_src, average_per_proto_dst = {}, {}
        average_per_proto_src_port, average_per_proto_dst_port = {}, {}
        
        columns = self.columns
        base_row = {c: [] for c in columns}
        
        start_time = time.time()
        total_duration = 0
        first_packet_time = None
        last_packet_time = 0
        count = 0
        count_rows = 0
        
        try:
            f = open(pcap_file, 'rb')
            pcap = dpkt.pcap.Reader(f)
            scapy_pak = rdpcap(pcap_file)
        except Exception as e:
            print(f"[ERROR] Impossible de lire le fichier PCAP : {e}")
            return False
        
        print(f"[INFO] Nombre total de paquets à traiter : {len(scapy_pak)}")
        
        for ts, buf in tqdm(pcap, desc="Traitement des paquets"):
            if count >= len(scapy_pak):
                break
                
            # Gestion des paquets Bluetooth/Zigbee avec Scapy
            try:
                if hasattr(scapy_pak[count], 'bluetooth'):
                    count += 1
                    continue
                elif 'ZigbeeNWK' in str(type(scapy_pak[count])):
                    zigbee = Communication_zigbee(scapy_pak[count])
                    count += 1
                    continue
            except:
                pass
            
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                count += 1
            except:
                count += 1
                continue
            
            # Initialisation du temps de référence
            if first_packet_time is None:
                first_packet_time = ts
            
            # Calcul de l'IAT (Inter-Arrival Time)
            IAT = ts - last_packet_time if last_packet_time > 0 else 0
            last_packet_time = ts
            
            ethernet_frame_size = len(buf)
            ethernet_frame_type = eth.type
            total_duration = ts - first_packet_time if first_packet_time else 0
            
            # Initialisation des variables pour ce paquet
            src_port, src_ip, dst_port, dst_ip = 0, "0.0.0.0", 0, "0.0.0.0"
            time_to_live, header_len, proto_type, protocol_name = 0, 0, 0, "UNKNOWN"
            flow_duration, flow_byte = 0, 0
            src_byte_count, dst_byte_count = 0, 0
            src_pkts, dst_pkts = 0, 0
            connection_status, number = 0, 1
            src_to_dst_pkt, dst_to_src_pkt = 0, 0
            src_to_dst_byte, dst_to_src_byte = 0, 0
            
            # Flags TCP
            flag_values = [0] * 8  # Initialiser avec 8 valeurs
            ack_count, syn_count, fin_count, urg_count, rst_count = 0, 0, 0, 0, 0
            
            # Protocoles (initialisation à 0)
            udp = tcp = http = https = arp = smtp = irc = ssh = dns = ipv = icmp = igmp = 0
            telnet = dhcp = llc = mac = rarp = mqtt = coap = 0
            
            # Features statistiques
            sum_packets = min_packets = max_packets = mean_packets = std_packets = 0
            magnitude = radius = correlation = covariance = var_ratio = weight = 0
            idle_time = active_time = 0
            
            # Features WiFi
            type_info = sub_type_info = ds_status = src_mac = dst_mac = 0
            sequence = pack_id = fragments = wifi_dur = 0
            
            # Traitement selon le type Ethernet
            if eth.type == dpkt.ethernet.ETH_TYPE_IP:
                ipv = 1
                ip = eth.data
                
                # Ignorer IPv6
                if isinstance(ip, dpkt.ip6.IP6):
                    continue
                
                try:
                    con_basic = Connectivity_features_basic(ip)
                    src_ip = con_basic.get_source_ip()
                    dst_ip = con_basic.get_destination_ip()
                    proto_type = con_basic.get_protocol_type()
                    protocol_name = get_protocol_name(proto_type)
                    
                    ips.add(dst_ip)
                    ips.add(src_ip)
                    
                    # Time features
                    con_time = Connectivity_features_time(ip)
                    time_to_live = con_time.time_to_live()
                    
                    # Bytes counting
                    conn_flags_bytes = Connectivity_features_flags_bytes(ip)
                    src_byte_count, dst_byte_count = conn_flags_bytes.count(src_ip_byte, dst_ip_byte)
                    
                    # Layer 3 features
                    potential_packet = ip.data
                    l_three = L3(potential_packet)
                    udp = l_three.udp()
                    tcp = l_three.tcp()
                    
                    # Protocol specific flags
                    if protocol_name == "ICMP":
                        icmp = 1
                    elif protocol_name == "IGMP":
                        igmp = 1
                    
                    # Layer 1 features
                    l_one = L1(potential_packet)
                    llc = l_one.LLC()
                    mac = l_one.MAC()
                    
                    # Packet counting
                    if src_ip not in src_packet_count:
                        src_packet_count[src_ip] = 0
                    src_packet_count[src_ip] += 1
                    
                    if dst_ip not in dst_packet_count:
                        dst_packet_count[dst_ip] = 0
                    dst_packet_count[dst_ip] += 1
                    
                    src_pkts = src_packet_count[src_ip]
                    dst_pkts = dst_packet_count[dst_ip]
                    
                    # Protocol counting
                    calculate_packets_counts_per_ips_proto(average_per_proto_src, protocol_name, 
                                                         src_ip, average_per_proto_dst, dst_ip)
                    
                    # UDP specific processing
                    if isinstance(potential_packet, dpkt.udp.UDP):
                        src_port = con_basic.get_source_port()
                        dst_port = con_basic.get_destination_port()
                        header_len = 8  # Fixed UDP header length
                        
                        # Layer 4 features
                        l_four = L4(src_port, dst_port)
                        l_two = L2(src_port, dst_port)
                        dhcp = l_two.dhcp()
                        dns = l_four.dns()
                        coap = l_four.coap()
                        
                        # Flow processing
                        flow = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)]))
                        flow_data = {
                            'byte_count': len(eth),
                            'header_len': header_len,
                            'ts': ts
                        }
                        
                        if flow not in udpflows:
                            udpflows[flow] = []
                        udpflows[flow].append(flow_data)
                        
                        # Flow statistics
                        flow_byte, flow_duration, max_duration, min_duration, sum_duration, \
                        average_duration, std_duration, idle_time, active_time = get_flow_info(udpflows, flow)
                        
                        src_to_dst_pkt, dst_to_src_pkt, src_to_dst_byte, dst_to_src_byte = \
                        get_src_dst_packets(udpflows, flow)
                    
                    # TCP specific processing
                    elif isinstance(potential_packet, dpkt.tcp.TCP):
                        src_port = con_basic.get_source_port()
                        dst_port = con_basic.get_destination_port()
                        header_len = con_basic.get_header_len()
                        
                        # Flags processing
                        flag_values = get_flag_values(ip.data)
                        if len(flag_values) < 8:
                            flag_values.extend([0] * (8 - len(flag_values)))
                        
                        # Layer 4 features
                        l_four = L4(src_port, dst_port)
                        http = l_four.http()
                        https = l_four.https()
                        ssh = l_four.ssh()
                        irc = l_four.IRC()
                        smtp = l_four.smtp()
                        mqtt = l_four.mqtt()
                        telnet = l_four.telnet()
                        
                        # HTTP status
                        try:
                            http_info = dpkt.http.Response(ip.data)
                            connection_status = http_info.status
                        except:
                            connection_status = 0
                        
                        # Flow processing
                        flow = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)]))
                        flow_data = {
                            'byte_count': len(eth),
                            'header_len': header_len,
                            'ts': ts
                        }
                        
                        # Flags counting
                        ack_count, syn_count, fin_count, urg_count, rst_count = \
                        compare_flow_flags(flag_values, ack_count, syn_count, fin_count, urg_count, rst_count)
                        
                        if flow not in tcpflows:
                            tcpflows[flow] = []
                        tcpflows[flow].append(flow_data)
                        
                        # Flow statistics
                        flow_byte, flow_duration, max_duration, min_duration, sum_duration, \
                        average_duration, std_duration, idle_time, active_time = get_flow_info(tcpflows, flow)
                        
                        src_to_dst_pkt, dst_to_src_pkt, src_to_dst_byte, dst_to_src_byte = \
                        get_src_dst_packets(tcpflows, flow)
                
                except Exception as e:
                    print(f"[WARNING] Erreur lors du traitement du paquet IP {count}: {e}")
                    continue
            
            elif eth.type == dpkt.ethernet.ETH_TYPE_ARP:
                protocol_name = "ARP"
                arp = 1
                if protocol_name not in packets_per_protocol:
                    packets_per_protocol[protocol_name] = 0
                packets_per_protocol[protocol_name] += 1
            
            elif eth.type == dpkt.ethernet.ETH_TYPE_REVARP:
                rarp = 1
            
            # Calcul des taux
            rate = srate = drate = 0
            if flow_duration > 0:
                rate = len(tcpflows.get(flow, [])) / flow_duration if 'flow' in locals() else 0
                srate = src_to_dst_pkt / flow_duration
                drate = dst_to_src_pkt / flow_duration
            
            # Calcul des features dynamiques (toutes les 20 paquets)
            ethsize.append(ethernet_frame_size)
            if len(ethsize) >= 20:
                try:
                    dy = Dynamic_features()
                    # Calculs statistiques de base
                    sum_packets = sum(ethsize)
                    min_packets = min(ethsize)
                    max_packets = max(ethsize)
                    mean_packets = np.mean(ethsize)
                    std_packets = np.std(ethsize)
                    
                    # Features géométriques (simplifiées)
                    if len(ethsize) > 1:
                        magnitude = np.sqrt(sum([x**2 for x in ethsize]))
                        radius = max(ethsize) - min(ethsize)
                        covariance = np.var(ethsize)  # Approximation
                        weight = sum(ethsize) / len(ethsize)
                    
                    ethsize = []  # Reset
                except Exception as e:
                    print(f"[WARNING] Erreur calcul features dynamiques: {e}")
            
            # Construction de la ligne de données
            new_row = {
                "flow_duration": flow_duration,
                "Header_Length": header_len,
                "Protocol Type": proto_type,
                "Duration": total_duration,
                "Rate": rate,
                "Srate": srate,
                "Drate": drate,
                "fin_flag_number": flag_values[0],
                "syn_flag_number": flag_values[1],
                "rst_flag_number": flag_values[2],
                "psh_flag_number": flag_values[3],
                "ack_flag_number": flag_values[4],
                "ece_flag_number": flag_values[6] if len(flag_values) > 6 else 0,
                "cwr_flag_number": flag_values[7] if len(flag_values) > 7 else 0,
                "ack_count": ack_count,
                "syn_count": syn_count,
                "fin_count": fin_count,
                "urg_count": urg_count,
                "rst_count": rst_count,
                "HTTP": http,
                "HTTPS": https,
                "DNS": dns,
                "Telnet": telnet,
                "SMTP": smtp,
                "SSH": ssh,
                "IRC": irc,
                "TCP": tcp,
                "UDP": udp,
                "DHCP": dhcp,
                "ARP": arp,
                "ICMP": icmp,
                "IPv": ipv,
                "LLC": llc,
                "Tot sum": sum_packets,
                "Min": min_packets,
                "Max": max_packets,
                "AVG": mean_packets,
                "Std": std_packets,
                "Tot size": ethernet_frame_size,
                "IAT": IAT,
                "Number": number,
                "Magnitue": magnitude, 
                "Radius": radius,
                "Covariance": covariance,
                "Variance": std_packets**2 if std_packets > 0 else 0,
                "Weight": weight,
                "label": f"{category}-{attack_type}-{protocol}",  # À définir selon les besoins
                "cumulative_duration": total_duration,
                "remaining_time": f"{total_duration - flow_duration}"  # À calculer selon les besoins
            }
            
            # Ajout à base_row avec vérification
            for c in columns:
                if c in new_row:
                    base_row[c].append(new_row[c])
                else:
                    print(f"[WARNING] Colonne manquante : {c}")
                    base_row[c].append(0)  # Valeur par défaut
            
            count_rows += 1
        
        # Fermeture du fichier
        f.close()
        
        print(f"[INFO] {count_rows} lignes traitées")
        
        # Création du DataFrame
        try:
            processed_df = pd.DataFrame(base_row)
            print(f"[INFO] DataFrame créé avec {len(processed_df)} lignes et {len(processed_df.columns)} colonnes")
            
            # Vérification des colonnes
            missing_cols = set(columns) - set(processed_df.columns)
            if missing_cols:
                print(f"[WARNING] Colonnes manquantes dans le DataFrame: {missing_cols}")
                for col in missing_cols:
                    processed_df[col] = 0
            
            # Réorganiser les colonnes dans l'ordre souhaité
            processed_df = processed_df[columns]
            
            # Sauvegarde
            output_file = f"{csv_file_name}.csv"
            processed_df.to_csv(output_file, index=False)
            print(f"[SUCCESS] Fichier sauvé : {output_file}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Erreur lors de la création du DataFrame : {e}")
            return False
