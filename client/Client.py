import zmq
import cv2 as cv
import json
import numpy as np
# import timeit as time

#ClientRequestGetImage 0
#ClientRequestDefault 1

class CClient:
    def __init__(self, serverIp, serverPort):
        self.m_ip = serverIp
        self.m_port = serverPort
        self.m_context = zmq.Context()
        self.m_socket = self.m_context.socket(zmq.PAIR)
        self.m_isConnected = False

    def EXT_ConnectToServer(self):
        #recive time out beallitas
        #m_socket.setsockopt(ZMQ_RCVTIMEO, &(data->m_serversInformations.m_waitingBlockingTime), sizeof(data->m_serversInformations.m_waitingBlockingTime));

        connectString = 'tcp://' + self.m_ip + ':' + str(self.m_port)

		
        print ('Connecting to server: ' + connectString + '...')
        self.m_socket.connect(connectString)

        #if (self.m_socket.connected()):
        #    self.m_isConnected = True
        #    print ('Connected to server: ' + self.m_ip)
        #else:
        #    print ("Canno't connect to server: " + self.m_ip)

    def EXT_GetImg(self, showRunTime):
        #message += "\0";
        #zmq::message_t zmqMessage(message.size() + 1);
        #memcpy(zmqMessage.data(), message.c_str(), message.size() + 1);
        
        #m_socket.
        #print ('111')
        # startTime = time.default_timer()
        self.m_socket.send(b'{"request":0}')
        #print ('222')
        message = self.m_socket.recv()
        # endTime = time.default_timer()
        # if (showRunTime):
        #     print ('Recive image time: ', "{:.4f}".format(endTime - startTime), ' sec')
        #print ('333')
        #print (message)

        jsonString = '';

        # startTime = time.default_timer()
        separate = 0
        for i in range(len(message)):
            if ((separate == 0) and (chr(message[i]) == '\0')):
                separate = i
                #print (i)

        #print ('Separate: ' + str(separate))
        for i in range(separate):
            jsonString = jsonString + chr(message[i])
            #print (chr(message[i]))
            #if (i < 20):
                #print (chr(message[i]))


        jsonObj = json.loads(jsonString)
        imgdata = message[separate+1:len(message)]
        img = cv.imdecode(np.fromstring(imgdata, dtype=np.uint8), -1)
        # endTime = time.default_timer()
        # if (showRunTime):
        #     print ('Decode image time: ', "{:.4f}".format(endTime - startTime), ' sec')
        #     print ('--------------------------')
        return img
        
#for (int i = 0; i < request.size(); ++i)
#{
#	//std::cout << ((char*)(message))[i];
#	if (((char*)(message))[i] == '\0')
#	{
#		separate = i;
#		i = request.size();
#	}
#}

#    def printhello(self):
#        print ('hello')
