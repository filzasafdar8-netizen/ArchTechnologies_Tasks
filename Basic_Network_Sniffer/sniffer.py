from scapy.all import sniff, IP, TCP, UDP

def packet_handler(packet):
    if packet.haslayer(IP):
        print("\nNew Packet Captured")
        print("Source IP:", packet[IP].src)
        print("Destination IP:", packet[IP].dst)

        if packet.haslayer(TCP):
            print("Protocol: TCP")
            print("Source Port:", packet[TCP].sport)
            print("Destination Port:", packet[TCP].dport)

        elif packet.haslayer(UDP):
            print("Protocol: UDP")
            print("Source Port:", packet[UDP].sport)
            print("Destination Port:", packet[UDP].dport)

print("Starting Network Sniffer...")
sniff(prn=packet_handler, count=10)