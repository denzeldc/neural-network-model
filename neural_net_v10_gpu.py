"""
Neural Network
Course: ECE 4600
Group 20

"""

import csv
import sklearn.metrics
import torch
import torch.nn as nn
import torch.cuda as cuda
import torchvision.transforms as transform
import torch.optim as optim
import matplotlib.pyplot as plt
import time


device = 'cuda:0'


# Neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, num_of_nodes):
        super().__init__()
        self.latency = nn.Linear((num_of_nodes * 5 + num_of_nodes ** 2), 30)  # 150,30
        self.adjacency = nn.Linear(num_of_nodes ** 2, 30)  # 100,30
        self.concat = nn.Linear(60, num_of_nodes ** 2)  # 60,30
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    # Normalize output data from the latency, and adjacency networks
    def normalize(self, input_data):
        size = len(input_data)
        batch_norm = nn.BatchNorm1d(10)
        batch_norm.to(device)
        normalized_data = batch_norm(torch.reshape(input_data, [size//10, 10]))
        normalized_data = torch.reshape(normalized_data, [size])
        
        return normalized_data
        

    def forward(self, latency_data, adjacency_data):
        latency = self.latency(latency_data)
        # between two successive layers, use ReLU
        latency = self.relu(latency)
        latency = self.normalize(latency)
        
        adjacency = self.adjacency(adjacency_data)
        adjacency = self.relu(adjacency)
        adjacency = self.normalize(adjacency)
        
        concatenated_data = torch.cat([latency, adjacency])
        output = self.concat(concatenated_data)
        output = self.sigmoid(output+adjacency_data)
        #output = self.sigmoid(output)
        
        # returned variables
        # output   - the output with the data set as either 1 or 0, converted
        #            into a difference matrix
        return output



def difference(output, adjacency_data):
    
    difference_matrix = output.clone()
    difference_matrix.cuda()
            
    for x in range(0, len(output)):
        if output[x] <= 0.5:
            difference_matrix[x] = 0
        else:
            difference_matrix[x] = 1
    
    for x in range(0, len(output)):
        if adjacency_data[x] == 0 and difference_matrix[x] == 1:
            difference_matrix[x] = 0  # convert to 0; networks don't gain connections after a link cut
    
    # create difference matrix; same entries in a given index are 0, different entries in given index are 1s
    for x in range(0, len(output)):
        if adjacency_data[x] != difference_matrix[x]:
            difference_matrix[x] = 1
        else:
            difference_matrix[x] = 0
    
        
    return difference_matrix



# createMatrix()
# Returns a matrix from a 1D list
def createMatrix(data, num_of_nodes, start, end):
    nxn_matrix = []
    temp_list = []

    for x in range(start, end):

        temp_list.append(data[x])  # add data to the list

        if (len(temp_list) == num_of_nodes):
            nxn_matrix.append(temp_list)  # add the row of entries to the matrix
            temp_list = []  # clear list to create the next row

    return nxn_matrix



# separate the data in the csv file into its components and store them in
# their respective lists
def decodeData(data, adj_mat_bef_list, adj_mat_aft_list, latency_matrix):
    # separate the data
    for y in range(0, len(data)):
        network_data = data[y][0]
        SIMULATION_COUNT = int(network_data[0])
        TIME_BETWEEN_PINGS = int(network_data[1])
        PINGS_PER_WINDOW = int(network_data[2])
        WINDOW_COUNT = int(network_data[3])
        LINK_CUT_WINDOW = int(network_data[4])
        MAX_TRAFFIC_DURATION = int(network_data[5])
        for x in range(1, len(data[y])):
            network_data = data[y][x]
            node_count = int(network_data[0])
            central_node_index = int(network_data[1])
            pf_length = int(network_data[2]) // 2

            alarmStart = 5
            latencyMatrixBeforeLinkCutStart = alarmStart + node_count
            latencyMatrixAfterLinkCutStart = latencyMatrixBeforeLinkCutStart + (node_count * LINK_CUT_WINDOW)

            # Start point for the adjacent node data to be read from csv
            possibleFaultsStart = latencyMatrixAfterLinkCutStart + (node_count * (WINDOW_COUNT - LINK_CUT_WINDOW))
            adjMatrixStart = possibleFaultsStart + (pf_length * 2)
            failureAdjMatrixStart = adjMatrixStart + (node_count ** 2)

            # 1 for alarm, 0 for no alarm for each node
            alarm_list = network_data[alarmStart:latencyMatrixBeforeLinkCutStart]
            alarm_list = list(map(int, alarm_list))
            latency_matrix_before_link_cut_flat = network_data[latencyMatrixBeforeLinkCutStart:
                                                               latencyMatrixAfterLinkCutStart]
            latency_matrix_before_link_cut_flat = list(map(float, latency_matrix_before_link_cut_flat))
            latency_matrix_after_link_cut_flat = network_data[latencyMatrixAfterLinkCutStart:possibleFaultsStart]
            latency_matrix_after_link_cut_flat = list(map(float, latency_matrix_after_link_cut_flat))
            # Creating flat list of adjacent node data
            possible_faults_flat = network_data[possibleFaultsStart:adjMatrixStart]
            possible_faults_flat = list(map(float, possible_faults_flat))
            print(possible_faults_flat)

            adj_matrix_flat = network_data[adjMatrixStart:failureAdjMatrixStart]
            adj_matrix_flat = list(map(int, adj_matrix_flat))
            failure_adj_matrix_flat = network_data[failureAdjMatrixStart:]
            failure_adj_matrix_flat = list(map(int, failure_adj_matrix_flat))

            latency_matrix_before_link_cut = []
            latency_matrix_after_link_cut = []
            adj_matrix = []  # failure free matrix
            failure_adj_matrix = []  # matrix with the failure
            possible_faults = []

            i = 0
            print(len(latency_matrix_before_link_cut_flat))
            for j in range(LINK_CUT_WINDOW):
                latency_matrix_before_link_cut.append([])
                for _ in range(node_count):
                    latency_matrix_before_link_cut[j].append(latency_matrix_before_link_cut_flat[i])
                    i += 1
            i = 0

            for j in range(WINDOW_COUNT - LINK_CUT_WINDOW):
                latency_matrix_after_link_cut.append([])
                for _ in range(node_count):
                    latency_matrix_after_link_cut[j].append(latency_matrix_after_link_cut_flat[i])
                    i += 1
            i = 0

            for j in range(node_count):
                adj_matrix.append([])
                for _ in range(node_count):
                    adj_matrix[j].append(adj_matrix_flat[i])
                    i += 1
            i = 0
            for j in range(node_count):
                failure_adj_matrix.append([])
                for _ in range(node_count):
                    failure_adj_matrix[j].append(failure_adj_matrix_flat[i])
                    i += 1
            # Format for possible_faults needs to be the following:
            # [[central_node_name, adj_switch],[latency, adj_name1],[latency, adj_name2]]
            i = 0
            # print(possible_faults_flat)
            ADJ_DATA = 2
            for j in range(0, pf_length):
                possible_faults.append([])
                for _ in range(ADJ_DATA):
                    possible_faults[j].append(possible_faults_flat[i])
                    i += 1

            for j in range(1, len(possible_faults)):
                possible_faults[j][0] = float(possible_faults[j][0])

            neighbour_matrix_l = []
            for j in range(node_count):
                neighbour_matrix_l.append([])
                for _ in range(node_count):
                    neighbour_matrix_l[j].append(-100.0)
            nIndexList = []
            for a in range(0, (len(possible_faults))):
                if (int(100 * possible_faults[a][0])) == (int(100 * possible_faults[a][1])):
                    nIndexList.append(a)
            nIndexList.append(len(possible_faults_flat))
            nList = []
            for a in range(len(nIndexList)):
                if a + 1 < len(nIndexList):
                    nList.append(possible_faults[nIndexList[a]:nIndexList[a + 1]])
            for pf in nList:
                for node in pf:
                    neighbour_matrix_l[int(pf[0][0])][int(node[1])] = node[0]
                    neighbour_matrix_l[int(node[1])][int(pf[0][0])] = node[0]
            neighbour_matrix_m = []
            for pf in neighbour_matrix_l:
                neighbour_matrix_m += pf

            # How many nodes are there?
            num_of_nodes = int(data[y][x][0])
            # Data such as switch, central node index, and pf_length (possible fault length)
            misc_data = data[y][x][0:5]

            # Are there any disconnected nodes?
            # Not immediately relevant to machine learning, but saved
            # disc_node_data = data[y][x][len(misc_data):num_of_nodes+len(misc_data)]

            # get the adjacency matrices

            # get the latency data
            latency_data = latency_matrix_before_link_cut_flat + latency_matrix_after_link_cut_flat

            # gather the latency data from each simulation into a list
            latency_matrix.append(latency_data)
            neighbour_matrix.append(neighbour_matrix_m)

            # adj_mat_start_index = len(misc_data) + len(disc_node_data) + len(latency_data)

            # form the adjacency matrices before and after the link cut
            adj_mat_bef = adj_matrix
            adj_mat_aft = failure_adj_matrix

            # add the matrices to the list
            adj_mat_bef_list.append(adj_mat_bef)
            adj_mat_aft_list.append(adj_mat_aft)



# separate the latency data from the raw data
def decodeLatencyData(latency_matrix, num_of_nodes):
    lat_mat_list = []
    for x in range(0, len(latency_matrix)):
        latency_times = latency_matrix[x][0:num_of_nodes * 5]

        lat_mat_list.append(latency_times)

    return lat_mat_list



# load data from CSV files
def loadData(path):
    data = []
    for x in range(0, FILE_NUMBER):
        for y in range(0, DATA_THREAD_COUNT):
            try:
                with open(path + "/dataWLatency" + str(x) + "t" + str(y) + ".csv", "r") as f:

                    print("/dataWLatency" + str(x) + "t" + str(y) + ".csv")
                    reader = csv.reader(f)
                    data.append(list(reader))
            except FileNotFoundError:
                continue
    return data



#-----------------------------------------------------------------------------
# CONSTANTS

# Number of nodes in a network
NODE_10 = 10
NODE_30 = 30
NODE_60 = 60

# number of entries in an adjacency matrix given N nodes
ADJ_MAT_10_SIZE = 100   
ADJ_MAT_30_SIZE = 900
ADJ_MAT_60_SIZE = 3600

# number of latencies for each node
LATENCY_PER_NODE = 5

# number of files to retrieve from the directory
FILE_NUMBER = 1300

# number of threads that created simulation data
DATA_THREAD_COUNT = 8


# Directory of the .csv files. Change the directory depending on what folder directory
# you are storing your .csv simulation files

MODEL_PATH = "C:/Users/Denzel/Documents/Python Scripts/10-Node"

# 10-node data directory
node_10_path = "C:/Users/Denzel/Documents/Python Scripts/10-Node"

# 30-node data directory
node_30_path = "C:/Users/Denzel/Documents/Python Scripts/30-Node"

# 60-node data directory
node_60_path = "C:/Users/Denzel/Documents/Python Scripts/60-Node"


node_path = [node_10_path, node_30_path, node_60_path]
num_of_nodes = [NODE_10, NODE_30, NODE_60]
latency_total_size = [LATENCY_PER_NODE*x for x in num_of_nodes]
adj_total_size = [ADJ_MAT_10_SIZE, ADJ_MAT_30_SIZE, ADJ_MAT_60_SIZE]


# change the int value to change the node network to use to train model
# 0 - 10-Node Networks
# 1 - 30-Node Networks
# 2 - 60-Node Networks
node_network_to_test = 0



node_data = []
adj_mat_bef_list = []
adj_mat_aft_list = []
latency_matrix = []
neighbour_matrix = []

# get data of 30-node networks
raw_data = loadData(node_path[node_network_to_test])

# get the adjacency matrices and the latency matrices
decodeData(raw_data, adj_mat_bef_list, adj_mat_aft_list, latency_matrix)

print("\nAdjacency matrix (before link cut) stats:")
print(adj_mat_bef_list[0])
print(len(adj_mat_bef_list))
print(len(adj_mat_bef_list[0]))
print(len(adj_mat_bef_list[0][0]))

print("\nAdjacency matrix (after link cut) stats:")
print(adj_mat_aft_list[0])
print(len(adj_mat_aft_list))
print(len(adj_mat_aft_list[0]))
print(len(adj_mat_aft_list[0][0]))

print("\nLatency data stats:")
print(latency_matrix[0])
print(len(latency_matrix))
print(len(latency_matrix[0]))

# get the latency data
final_latency_matrix = decodeLatencyData(latency_matrix, num_of_nodes[node_network_to_test])


# convert the latency data into floating point numbers
for x in range(0, len(final_latency_matrix)):
    for y in range(0, len(final_latency_matrix[x])):
        final_latency_matrix[x][y] = float(final_latency_matrix[x][y])
for x in range(0, len(neighbour_matrix)):
    for y in range(0, len(neighbour_matrix[x])):
        neighbour_matrix[x][y] = float(neighbour_matrix[x][y])

print("Neighbour matrix")
for matrix in neighbour_matrix:
    print("%d" % (len(matrix)))
    
    
# 30x5 matrix adjacency matrix
latency_list_2D = []
neighbour_matrix_2D = []
for x in range(0, len(final_latency_matrix)):
    temp = torch.FloatTensor(createMatrix(final_latency_matrix[x], num_of_nodes[node_network_to_test], 0, len(final_latency_matrix[x]))).cuda()
    latency_list_2D.append(temp)
for x in range(0, len(neighbour_matrix)):
    temp = torch.FloatTensor(createMatrix(neighbour_matrix[x], num_of_nodes[node_network_to_test], 0, len(neighbour_matrix[x]))).cuda()
    neighbour_matrix_2D.append(temp)

print("\n2D latency list")
print(latency_list_2D[0])
print(len(latency_list_2D))
print(len(latency_list_2D[0]))
print(len(latency_list_2D[0][0]))

latency_list_tensor = []
for x in range(0, len(final_latency_matrix)):
    latency_list_tensor.append(torch.FloatTensor(final_latency_matrix[x] + neighbour_matrix[x]).cuda())

adj_list_bef = []  # the 1D adjacency list before the link cut (1x900)
adj_list_aft = []  # the 1D adjacency list after the link cut (1x900)
temp = []
temp_aft = []


# get 1x900 1D tensor for the adjacency matrix
for x in range(0, len(adj_mat_bef_list)):
    for y in range(0, len(adj_mat_bef_list[x])):
        for z in range(0, len(adj_mat_bef_list[x][y])):
            temp.append(float(adj_mat_bef_list[x][y][z]))
            temp_aft.append(float(adj_mat_aft_list[x][y][z]))
    adj_list_bef.append(torch.FloatTensor(temp).cuda())
    adj_list_aft.append(torch.FloatTensor(temp_aft).cuda())
    temp = []
    temp_aft = []
    


# 30 x 30 adjacency matrix (for 30-node networks)
adj_list_bef_2D = []
for x in range(0, len(adj_list_bef)):
    # temp = cuda.FloatTensor(createMatrix(adj_list_bef[x], NODE_30, 0, len(adj_list_bef[x])))
    temp = torch.reshape(adj_list_bef[x], (num_of_nodes[node_network_to_test], num_of_nodes[node_network_to_test])).cuda()
    #temp.to(device)
    adj_list_bef_2D.append(temp)

print("\nadj_list_bef")
print(adj_list_bef[0])
print("\nadj_list_bef_2D")
print(adj_list_bef_2D[0])
print(len(adj_list_bef_2D[0]))
print(len(adj_list_bef_2D[0][0]))

print(latency_list_2D[0])
print(len(latency_list_2D))
print(len(latency_list_2D[0]))
print(len(latency_list_2D[0][0]))


batch_norm = nn.BatchNorm1d(10).cuda()
norm_latency_list_2D = []
norm_latency_list = []

norm_row = num_of_nodes[node_network_to_test] * 5 + num_of_nodes[node_network_to_test] ** 2

for x in range(0, len(latency_list_tensor)):
    norm_latency = batch_norm(torch.reshape(latency_list_tensor[x].cuda(), [norm_row//10, 10]))
    norm_latency_list_2D.append(norm_latency)
    norm_latency_list.append(torch.reshape(norm_latency, [norm_row]).cuda())

print("2D Normalized latency list")
print(norm_latency_list_2D[0])
print(len(norm_latency_list_2D[0]))
print(len(norm_latency_list_2D[0][0]))
print("1D Normalized latency list")
print(norm_latency_list[0])
print(norm_latency_list[0][0])
print(len(norm_latency_list[0]))

print("Latency list tensor")
print(latency_list_tensor[0])
print(len(latency_list_tensor))
print(len(latency_list_tensor[0]))

difference_matrix_list = []  # a list of difference matrices
difference_matrix = []  # an element in the difference matrix list
for x in range(0, len(adj_list_bef)):
    
    difference_matrix = adj_list_bef[x].cpu() != adj_list_aft[x].cpu()
    difference_matrix_list.append(torch.FloatTensor(difference_matrix.float()))
    difference_matrix = []

    # difference = torch.logical_not(torch.logical_and(adj_list_bef[x], adj_list_aft[x]))
    # difference_matrix.append(difference.float())

print("Difference matrix")
print(difference_matrix_list[0])

# print(print(len(neighbour_matrix[0][0])))
# print(neighbour_matrix_2D)

############################ NEURAL NETWORK ##################################

#                               INPUTS
# latency_list_tensor (1D array of latency data (dimensions: (1,50)))
# adj_list_bef        (1D array of adjacency data (dimensions: (1,100)))
# latency_list_2D     (2D array of latency data (dimensions: (5,10)))
# adj_list_bef_2D     (2d array of adjacency data (dimensions: (10,10)))
# norm_latency_list

start_time = time.time() # start recording time

# construct the neural network model
latency_net = NeuralNetwork(num_of_nodes[node_network_to_test])
latency_net.to(device)
loss_function = nn.BCEWithLogitsLoss()  # initialize loss function


# number of datasets in the test set for the model
test_data_length = 200

# stochastic gradient descent
optimizer = optim.SGD(latency_net.parameters(), lr=0.05)
train_prob_list = []
predicted_adj_list_train = []

# Train the model with a
latency_net.train()
for x in range(0, len(latency_list_tensor) - test_data_length):

    optimizer.zero_grad()  # zero the gradient
    
    # feed the test set
    out = latency_net.forward(norm_latency_list[x].cuda(), adj_list_bef[x].cuda()).cuda()
        
    pred_diff_matrix = difference(out, adj_list_bef[x])
    

    loss = loss_function(pred_diff_matrix, difference_matrix_list[x].cuda()).cuda()  # find the difference between predicted and actual link cut
    #loss = loss_function(out, difference_matrix_list[x].cuda())
    #loss = loss_function(out, adj_list_aft[x]) # find difference between probabilities and actual values
    
    loss.backward()  # backpropagation
    optimizer.step()
    
    
    # Print sample outputs from training set; comment out to disable
    if x % 10 == 0:
        print("\ndiff_out %d" % (x))
        print(out)
        print("adj_list_aft %d" % (x))
        #print("loss")
        #print(loss)
    
    predicted_adj_list_train.append(pred_diff_matrix)
    train_prob_list.append(out)

end_time = time.time()  #end recording time


# Testing accuracy using test set
test_start_index = len(latency_list_tensor) - test_data_length
test_end_index = len(latency_list_tensor)

# store the predicted adjacency matrices produced by the model
predicted_adj_list_test = []
output_list = []
test_prob_list = []


latency_net.eval()
# test the model
with torch.no_grad():
    for x in range(test_start_index, test_end_index):

        out = latency_net.forward(norm_latency_list[x], adj_list_bef[x])
        
        # get the predicted difference matrix
        test_pred_diff_matrix = difference(out, adj_list_bef[x])
        
        output_list.append(test_pred_diff_matrix.clone().detach())
        
        # ---------------------------------------------------------------------

        # compare the adjacency matrices for a sample test simulation
        if x == (test_start_index + test_end_index) // 2:
            print("\nAdjacency matrix for a network before link cut")
            print(adj_list_bef[x])
            print("\nAdjacency matrix for a network after link cut")
            print(adj_list_aft[x])
            print("\nDifference matrix between before and after matrices")
            print(difference_matrix_list[x])
            print("\nDifference matrix after link cut predicted by model")
            print(test_pred_diff_matrix)

        predicted_adj_list_test.append(test_pred_diff_matrix)
        test_prob_list.append(out)
    
    accuracy_list = []
    # test accuracy per test data
    for x in range(0, len(predicted_adj_list_test)):
        accuracy_matrix = torch.eq(predicted_adj_list_test[x], difference_matrix_list[x + test_start_index].cuda())
        num_of_matches = torch.sum(accuracy_matrix)
        accuracy_list.append(num_of_matches / adj_total_size[node_network_to_test])
        # print(accuracy_matrix)


#Calculate ROC for training set
#concatenate all of the probabilities in the list of predicted values into one tensor
pred_prob_train = torch.cat(train_prob_list,0)

#true_prob_train = torch.cat(adj_list_aft[:len(latency_list_tensor)-test_data_length], 0) # we are calculating for the ROC of the difference matrix, not the adjacency matrix
true_prob_train = torch.cat(difference_matrix_list[:len(latency_list_tensor)-test_data_length],0) # true training values are the true values after the link cut


pred_prob_train_np = pred_prob_train.cpu().detach().numpy()
true_prob_train_np = true_prob_train.cpu().detach().numpy()

auc_train = sklearn.metrics.roc_auc_score(true_prob_train_np, pred_prob_train_np)
fpr_train, tpr_train, thresholds_train = sklearn.metrics.roc_curve(true_prob_train_np, pred_prob_train_np)


# Calculate ROC for test set
pred_prob_test = torch.cat(test_prob_list,0)
    

#true_prob_test = torch.cat(adj_list_aft[-test_data_length:], 0) # we are calculating for the ROC of the difference matrix, not the adjacency matrix
true_prob_test = torch.cat(difference_matrix_list[-test_data_length:],0) # true training values are the true values after the link cut

# numpy version of probability scores
pred_prob_test_np = pred_prob_test.cpu().detach().numpy()
true_prob_test_np = true_prob_test.cpu().detach().numpy()

# calculate ROC curve parameters and AUC score
auc_test = sklearn.metrics.roc_auc_score(true_prob_test_np, pred_prob_test_np)
fpr_test, tpr_test, thresholds_test = sklearn.metrics.roc_curve(true_prob_test_np, pred_prob_test_np)



print("\n\nAccuracy of model at predicting adjacency matrices after link cut")
print(
    "\nThe following accuracies were obtained from a test set with 200 datasets of adjacency matrix data and latency data")
print(
    "The following shows how well the model predicts the adjacency matrices (accuracy for each adjacency matrix shown in decimals)\n")
print(accuracy_list)


print("\n\nROC statistics")

print("\nAUC score for the Training Set")
print(auc_train)

"""# Print Data for the roc curve parameters of the training set
print("\nFalse Positive Rate for the Training Set for each threshold")
print(fpr_train)

print("\nTrue Positive Rate for the Training Set for each threshold")
print(tpr_train)

print("\nThresholds for the Training Set")
print(thresholds_train)
"""

print("\nAUC score for the Test Set")
print(auc_test)

"""# Print Data for the roc curve parameters of the test set
print("\nFalse Positive Rate for the Test Set for each threshold")
print(fpr_test)

print("\nTrue Positive Rate for the Test Set for each threshold")
print(tpr_test)

#print("\nThresholds for the Test Set")
#print(thresholds_test)
"""


# thresholds change with each
print("\n\nThresholds train length")
print(len(thresholds_train))
print("Thresholds test length")
print(len(thresholds_test))


print(len(fpr_train))
print(len(tpr_train))
print(len(fpr_test))
print(len(tpr_test))


print("\n\n The time elapsed while training the model:")
print(end_time - start_time)

# Plot the ROC curves for the training and test set
roc_fig = plt.figure(1)
plt.plot(fpr_train, tpr_train)
plt.plot([0,1], [0,1], color="blue")
#plt.plot([0,1], [1,0], color="red")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0,1])
plt.ylim([0,1])
plt.title("ROC curve (Training set) for %d-Node Networks" % num_of_nodes[node_network_to_test])



plt.figure(2)
plt.plot(fpr_test, tpr_test)
plt.plot([0,1], [0,1], color="blue")
#plt.plot([0,1], [1,0], color="red")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0,1])
plt.ylim([0,1])
plt.title("ROC curve (Test set) for %d-Node Networks" % num_of_nodes[node_network_to_test])


f1_pred_train = torch.cat(predicted_adj_list_train,0)
f1_pred_train = f1_pred_train.int().cpu().detach().numpy()

f1_pred_test = torch.cat(predicted_adj_list_test,0)
f1_pred_test = f1_pred_test.int().cpu().detach().numpy()

f1_score_train = sklearn.metrics.f1_score(true_prob_train_np.astype(int), f1_pred_train.astype(int), pos_label=1)
f1_score_test = sklearn.metrics.f1_score(true_prob_test_np.astype(int), f1_pred_test.astype(int), pos_label=1)

print("\n\nF1 Score (training set)")
print(f1_score_train)

print("\n\nF1 Score (test set)")
print(f1_score_test)


#torch.save(latency_net.state_dict(), MODEL_PATH)
print("\nSaved Model to "+MODEL_PATH)
# print(output_list)
print("\n\nEnd of processing")

