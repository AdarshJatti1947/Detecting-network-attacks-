import socket

HOST = '127.0.0.1'
PORT = 8888

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    protocol_type = 'tcp'
    service = 'http_2784'
    flag = 'SH' 

    packet = f'{protocol_type},{service},{flag}'.encode()
    s.sendall(packet)
    data = s.recv(1024)

print('Received', repr(data.decode()))
