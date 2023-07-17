# By Namli & Nouri (13.04.2022)

# required libraries are imported
import datetime as dt
import random
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,roc_auc_score,balanced_accuracy_score

n0 = dt.datetime.now()

# determine the values of the required parameters
# define the step size for each layer
stepSize = [0.3, 0.1, 0.2]
# define the alpha value for momentum term
alfa = 0.2
# define the epsilon for stop criteria
epsilon = 0.1
# determine the values of the required parameters
numberOfClass = 2  # define the # of class
numberOfRun = 20  # define the # of run
numberOfIteration = 100  # define the # of iteration for training

# different coefficients (a) are assigned for sigmoid functions
a = [3, 1, 2]

pieces_testset = 10 # define the number of pieces for testset
sinifsayisi = numberOfClass    # define the # of class
percentage = 60
featureselectionnumber=10
Selected_Feature_Number = 5
durum=0

# determine the values of the required parameters for SOM
neuronnumber = 120
gridSizeX = 10
gridSizeY = 10
iterationnumber = 100
kosturmasayisi = 1
sigma_I = 3
landa = 2.5
L_I = 1

# define necessary variables
E_iterasyon = []
E_iterasyon_test = []
training_itr_number=[]
Accuracy_List = []
Accuracy_yavas_List = []
Precision_List = []
Recall_List = []
F_Measure_List = []
roc_auc_ovo_List=[]
roc_auc_ovr_List=[]
Specificity_List=[]
Balanced_accuracy_List=[]
specificity_for_class_1_List=[]
specificity_for_class_0_List=[]
EgitimSure = []
TestSure = []
TestSureSOM = []
EgitimSureSOM = []
Train_Index_Tut=[]
W_son_Tut=[]
c_son_Tut=[]
sigma_son_Tut=[]
TestSetS=[]
Features_index_const=[]
Potentialindex_ALL=[]
Potentialpredict_ALL=[]
predict_ALL=[]

# Import Data set
df = pd.read_csv('adult_ql.csv')  # (use "r" before the path string to address special character, such as '\'). Don't forget to put the file name at the end of the path + '.xlsx'
Input = df.to_numpy()  # Convert Data set to numpy(Data set include Feacture ID,Input Index and Class information)
numberofinput = len(Input[1:, :])  # there is how many number of sample in data set

# Define percentage function
def calc_percentage(a, b):
    return round((a * b) / 100, 0)


# Define function to split data set to training and test data set (20% for training rest for test)
def Generate_Training_Test_Set(Input):
    Numberof_Training_Set = int(calc_percentage(percentage, numberofinput))  # find number of 20% of data set for training
    Input_Index = list(range(1, (numberofinput+1)))   #list of input index
    Random_Training_Index = np.array(random.sample(Input_Index, Numberof_Training_Set))     #select random 20% for training set(according to Input Index)
    Random_Training_Index.sort()

    # Creat Training  set
    Training_Set1 = np.array(Input[Random_Training_Index.astype(int)])  # Training set without features ID(Create  Training Set according to random selected Index)
    Training_Set = np.vstack((Input[0, :], Training_Set1))  # Add features ID to Training set

    # Creat Test set
    Test_Set = np.array(np.delete(Input, Random_Training_Index,axis=0))  # Delete selected training set from input to create Test set (without Features ID)
    Test_Set_Index=list(Test_Set[1:,0]) #Test set Index
    NumberofSample_TestSet=len(Test_Set)
    return Training_Set, Random_Training_Index, Test_Set, Test_Set_Index, NumberofSample_TestSet

def Feature_Selection(Features_index_const,IN):
    if len(Features_index_const)==0:
        Features = list(np.unique(IN[0, 1:-sinifsayisi]))#find the unique features
        Random_Features = np.array(random.sample(Features, Selected_Feature_Number))#select 10 number of features randomly
        Random_Features.sort()

        SelecetedFeatures_index = []
        for h in Random_Features:
            index = np.argwhere(IN[0, 1:-sinifsayisi] == h)
            index = np.reshape(index, (1, len(index)))  # find the index of each randomly selected features
            SelecetedFeatures_index = np.append(SelecetedFeatures_index, index[0])
            SelecetedFeatures_index = SelecetedFeatures_index.astype(int)
    else:
        SelecetedFeatures_index=Features_index_const

    IN1 = IN[:, 1:-sinifsayisi]
    TestSet_With_Selectedindex1 = IN1[:,SelecetedFeatures_index]  # Create test set with random selected index ( without Input Index and class information with Features ID)
    TestSet_With_Selectedindex2 = np.vstack((TestSet_With_Selectedindex1.T, IN[:, -sinifsayisi:].T))  ## add  class information to last column of test with random selected index (data set without Input  Index and with class information and Feature ID)
    TestSet_With_Selectedindex3 = np.vstack((IN[:, 0].T,TestSet_With_Selectedindex2))  ## add Input Index to first column of test set(data set with selected features with Input  index and with class information and Feature ID)
    TestSet_With_Selectedindex = TestSet_With_Selectedindex3.T  # transpose
    dim_Som = len(SelecetedFeatures_index)
    return SelecetedFeatures_index, TestSet_With_Selectedindex, dim_Som

# assign random weights for each layer
def __randomWeights(range1, range2) -> list:
    return [[round(random.uniform(-0.25, 0.25), 1) for _ in range(range1)] for _ in range(range2)]

# define the sigmoid (logistic function) for each layer
def __sigmoid(x, a):
    return (1 / (1 + np.exp(-a * x)))

# derivative of sigmoid function for each layer
def __deriv(x, a):
    return (a * np.exp(-a * x) / ((1 + np.exp(-a * x)) ** 2))

# Location of neurons for SOM
Neuron_XY = np.zeros((neuronnumber, 2))
for i in range(neuronnumber):
    s = divmod(i, gridSizeX)
    Neuron_XY[i][0] = s[1] + 1
    Neuron_XY[i][1] = s[0] + 1

# distance between nerouns location for SOM
d_ij = np.zeros((neuronnumber, neuronnumber))
for i in range(0, neuronnumber):
    for j in range(0, neuronnumber):
        d_ij[i][j] = distance.euclidean(Neuron_XY[i,], Neuron_XY[j,])  # distance between nerouns

# training phase of MLP:
def mlp_training(Inputmatrix,Outputmatrix):

    n1 = dt.datetime.now()

    train_input_column = len(Inputmatrix[0])
    train_input_row = len(Inputmatrix)

    ns0 = numberOfClass
    ns1 = train_input_column + 2
    ns2 = train_input_column + 1

    E = []
    W0_son = []
    W0_tum = []
    W1_tum = []
    W2_tum = []
    E_instantaneous_training_run=[]

    for p in range(0,numberOfIteration):

        # Create weight matrices that will be used for the momentum term
        w1_old = __randomWeights(train_input_column + 1, ns1)
        w2_old = __randomWeights(ns1 + 1, ns2)
        w0_old = __randomWeights(ns2 + 1, ns0)

        # Create random weight matrices
        w1 = __randomWeights(train_input_column + 1, ns1)
        w2 = __randomWeights(ns1 + 1, ns2)
        w0 = __randomWeights(ns2 + 1, ns0)

        w1_list = [np.array(w1_old), np.array(w1)]
        w2_list = [np.array(w2_old), np.array(w2)]
        w0_list = [np.array(w0_old), np.array(w0)]

        # Training phase:
        for j in range(0, numberOfIteration):
            e_ani = []

            for i in range(0, train_input_row):
                # FIRST LAYER
                # add bayes term
                vektor1 = np.concatenate((Inputmatrix[i], [1]))
                V1 = np.dot(w1_list[1], vektor1)
                Y1 = __sigmoid(V1, a[1])

                # SECOND LAYER
                # add bayes term
                vektor2 = np.concatenate((Y1, [1]))

                V2 = np.dot(w2_list[1], vektor2)

                Y2 = __sigmoid(V2, a[2])

                # OUTPUT LAYER
                # add bayes term
                vektor0 = np.concatenate((Y2, [1]))
                V0 = np.dot(w0_list[1], vektor0)
                # the result of forward propagation
                Y0 = __sigmoid(V0, a[0])

                # calculate error term
                e = Outputmatrix[i] - Y0
                e_ani.append(0.5 * np.dot(e.T, e))

                # calculate local gradients
                d0 = e * __deriv(V0, a[0])

                # delete the column for bayes term from the weight matrix of output layer
                w0Del = np.delete(w0_list[1], ns2, axis=1)
                d2 = np.dot(w0Del.T, d0) * __deriv(V2, a[2])

                # delete the column for bayes term from the weight matrix of first layer

                w2Del = np.delete(w2_list[1], ns1, axis=1)

                d1 = np.dot(w2Del.T, d2) * __deriv(V1, a[1])

                # keep the weights before update
                W0_tut = w0_list[1]
                W2_tut = w2_list[1]
                W1_tut = w1_list[1]
                # keep updated weights as second item of list
                w0_list[1] = w0_list[1] + stepSize[0] * np.dot(d0[:, None], vektor0[None, :]) + alfa * (
                        w0_list[1] - w0_list[0])
                w2_list[1] = w2_list[1] + stepSize[2] * np.dot(d2[:, None], vektor2[None, :]) + alfa * (
                        w2_list[1] - w2_list[0])
                w1_list[1] = w1_list[1] + stepSize[1] * np.dot(d1[:, None], vektor1[None, :]) + alfa * (
                        w1_list[1] - w1_list[0])
                # keep the weights of one step before as the first item in the list
                w0_list[0] = W0_tut
                w2_list[0] = W2_tut
                w1_list[0] = W1_tut

            E.append(np.mean(e_ani))

            # calculate the mean error values for the relevant iteration
            if np.mean(e_ani) < epsilon:
                # if (max(E_ani) < epsilon):
                W0_son = np.array(w0_list[1])
                W2_son = np.array(w2_list[1])
                W1_son = np.array(w1_list[1])
                break

        training_itr_number.append(len(E))
        e_ani_egitim_deger = np.mean(E)

        # keep the latest updated weight matrix
        if len(W0_son) == 0:
            W0_son = w0_list[1]
            W2_son = w2_list[1]
            W1_son = w1_list[1]

        E_instantaneous_training_run.append(e_ani_egitim_deger)
        W0_tum.append(W0_son)
        W2_tum.append(W2_son)
        W1_tum.append(W1_son)

    #The weights of the model with the smallest error mean in the last iteration are used.
    w0_end = W0_tum[E_instantaneous_training_run.index(min(E_instantaneous_training_run))]
    w2_end = W2_tum[E_instantaneous_training_run.index(min(E_instantaneous_training_run))]
    w1_end = W1_tum[E_instantaneous_training_run.index(min(E_instantaneous_training_run))]

    n2 = dt.datetime.now()
    return W0_son, W2_son, W1_son, e_ani_egitim_deger, training_itr_number, n2, n1

# test phase of MLP:
def mlp_test(Inputtestmatrix, w0_son, w2_son, w1_son):

    n3 = dt.datetime.now()

    test_input_column = len(Inputtestmatrix[0])
    test_input_row = len(Inputtestmatrix)

    Yd_hesaplanan = []
    Test_Yd_hesaplanan_Sinif = []

    for i in range(0, test_input_row):
        # FIRST LAYER
        # add bayes term
        vektor1 = np.concatenate((Inputtestmatrix[i], [1]))

        V1 = np.dot(w1_son, vektor1)

        Y1 = __sigmoid(V1, a[1])

        # SECOND LAYER
        # add bayes term
        vektor2 = np.concatenate((Y1, [1]))

        V2 = np.dot(w2_son, vektor2)

        Y2 = __sigmoid(V2, a[2])

        # OUTPUT LAYER
        # add bayes term
        vektor0 = np.concatenate((Y2, [1]))

        V0 = np.dot(w0_son, vektor0)
        Y0 = __sigmoid(V0, a[0])

        # the result of back propagation
        for j in range(0, numberOfClass):
            if Y0[j] >= sum(Y0) - Y0[j]:
                Y0[j] = 1
            else:
                Y0[j] = 0

        Yd_hesaplanan.append(Y0)

    for k in range(0, test_input_row):
        if sum(Yd_hesaplanan[k]) == 1:
            for l in range(0, numberOfClass):
                if Yd_hesaplanan[k][l] == 1:
                    Test_Yd_hesaplanan_Sinif.append(l)
        else:
            Test_Yd_hesaplanan_Sinif.append(numberOfClass)

    n4 = dt.datetime.now()
    return Test_Yd_hesaplanan_Sinif

def SOM_training(neuronnumber, Input, dim_Som):
    Inputclass = []
    numberofsample = len(Input[0])
    Output=Input[1:,-sinifsayisi:]
    for k in range(0, numberofsample):
        for l in range(0, sinifsayisi):
            if Output[k][l] == 1:
                Inputclass.append(l)

    # Inputclass=Input[1:,-1]
    Input = Input[1:, 1:-sinifsayisi].T

    Egtim_suresi_1 = dt.datetime.now()
    w = np.array([[round(random.uniform(-0.25, 0.25), 4) for x in range(neuronnumber)] for y in
                  range(dim_Som)])  ##Initializing The Weights
    w_initial = w
    Winners_old = np.zeros(numberofsample)
    for n in range(1, iterationnumber + 1):  # beginning of each iteration
        Winners = np.zeros(numberofsample)
        for a in range(0, numberofsample):
            Distance = np.zeros((neuronnumber, numberofsample))
            for b in range(0, neuronnumber):
                Distance[b][a] = distance.euclidean(Input[:, a], w[:,
                                                                 b])  # calculate distance between each input and neurons by euclidean norm
            mindist = min(Distance[:, a])  # find minimum distance
            Winner = random.choice(np.argwhere(Distance[:, a] == np.min(Distance[:, a])))  # find index of winner neuron
            Winners[a] = Winner  # put winners in one vector for each input
            # Start updating
            sigma_N = sigma_I * np.exp(-n / (landa))  # the exponential decay function)
            H_n = np.exp(-(d_ij[Winner, :]) ** 2 / (2 * (sigma_N) ** 2))  # disp('H_n is the neighborhood function')
            H_nn = np.ones((dim_Som, 1)) * H_n
            L_n = L_I * np.exp(-n / landa)  # the learning rate parameter
            w_N = (w + (L_n * H_nn * (Input[:, a].reshape(dim_Som, 1) - w))).reshape(dim_Som,
                                                                                     neuronnumber)  # disp('w_N Is the updated weight vector')
            w_old = w  # keep old weights in w_old
            w = w_N  # replace old weights with updated weights
        # if (Winners_old == Winners).all():  # if winner in this step will be same as the previous step "Winners are same" will be printed
        # print("Winners are same")
        Winners_old = Winners
    Egtim_suresi_2 = dt.datetime.now()
    som_sinifbilgisi = np.zeros(numberofsample)
    obeklere_giren_veri_sayisi = np.zeros(neuronnumber)
    kac_sinifdan_veri = np.zeros(neuronnumber)
    obek_etiketi = np.zeros(neuronnumber)
    silinen_agliklar = []
    for l in range(0, neuronnumber):
        globals()[str(l + 1) + "obek"] = np.where(Winners == l)  # Test de l.obege hangi veriler var
        obeklere_giren_veri_sayisi[l] = np.sum(Winners == l)
        globals()[str(l + 1) + ".obege giren verilerin sinif bilgisi"] = np.zeros(
            len(globals()[str(l + 1) + "obek"][0]))
        if (obeklere_giren_veri_sayisi[l] != 0):
            for m in range(0, len(globals()[str(l + 1) + "obek"][0])):
                globals()[str(l + 1) + ".obege giren verilerin sinif bilgisi"][m] = Inputclass[
                    globals()[str(l + 1) + "obek"][0][m]]
            globals()[str(l + 1) + ".uniqueValues_sinifbilgisi"], occurCount_sinifbilgisi_test = np.unique(
                globals()[str(l + 1) + ".obege giren verilerin sinif bilgisi"], return_counts=True)
            kac_sinifdan_veri[l] = globals()[str(l + 1) + ".obekde kac sinfdan veri var"] = len(
                np.unique(globals()[str(l + 1) + ".obege giren verilerin sinif bilgisi"]))
            for v in range(0, sinifsayisi):
                if (globals()[str(l + 1) + ".uniqueValues_sinifbilgisi"] == v).all():
                    som_sinifbilgisi[globals()[str(l + 1) + "obek"]] = v
                    obek_etiketi[l] = v
                elif len(globals()[str(l + 1) + ".uniqueValues_sinifbilgisi"]) > 1:
                    som_sinifbilgisi[globals()[str(l + 1) + "obek"]] = sinifsayisi
                    obek_etiketi[l] = sinifsayisi
        else:
            kac_sinifdan_veri[l] = 0
            silinen_agliklar.append(l)  # index of clusters which are empty
            obek_etiketi[l] = sinifsayisi + 1
    EgitimSureSOM.append(Egtim_suresi_2 - Egtim_suresi_1)
    silinen_agliklar = np.array(silinen_agliklar)
    w_test = np.delete(w, silinen_agliklar, axis=1)  # delet weights which are not win
    obek_etiketi = obek_etiketi[obek_etiketi != sinifsayisi + 1]  # delet empty clusters
    neuronnumber_test = neuronnumber - len(
        silinen_agliklar)  # delet number of neuron whıch are not win from whole number of neuron
    return w_test, Winners, EgitimSureSOM, som_sinifbilgisi, obek_etiketi, silinen_agliklar, neuronnumber_test


def SOM_test(neuronnumber_test, Input_test, w_test, obek_etiketi):
    Input_test = Input_test[1:, :].T
    numberofinput_test = len(Input_test[0])
    Test_suresi_1 = dt.datetime.now()
    Winners_test = np.zeros(numberofinput_test)
    sinifbilgisi_test = np.zeros(numberofinput_test)
    for a in range(0, numberofinput_test):

        Distance_test = np.zeros((neuronnumber_test, numberofinput_test))
        for b in range(0, neuronnumber_test):
            Distance_test[b][a] = distance.euclidean(Input_test[:, a], w_test[:,
                                                                       b])  # calculate distance between each input and neurons by euclidean norm
        mindist_test = min(Distance_test[:, a])  # find minimum distance
        Winner_test = random.choice(np.argwhere(Distance_test[:, a] == np.min(Distance_test[:, a])))
        for v in range(0, sinifsayisi):
            sinifbilgisi_test[a] = obek_etiketi[Winner_test]
            # if obek_etiketi[Winner_test]== v:
            # sinifbilgisi_Test[numberofinput_test] = v
            # elif bek_etiketi[Winner_test]== v > 1:
            # sinifbilgisi_test[numberofinput_test] = sinifsayisi
        Winners_test[a] = Winner_test  # put wiiners in one vector for each input
    Test_suresi_2 = dt.datetime.now()
    TestSureSOM.append(Test_suresi_2 - Test_suresi_1)
    return sinifbilgisi_test

# calculate performance criteria
def performance(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif):
    confusionmatrix=confusion_matrix(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif)
    tp1, fn1, fp1, tn1 = confusion_matrix(Test_Yd_Sinif, Test_Yd_hesaplanan_Sinif).ravel()
    tn2, fp2, fn2, tp2 = confusion_matrix(Test_Yd_Sinif, Test_Yd_hesaplanan_Sinif).ravel()
    accuracy = round(accuracy_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif), 3)
    print("confusionmatrix",confusionmatrix)
    # Calculate accuracy for "yavas" class.
    sinifsayisinihai=sinifsayisi+1
    ConfusionMatrixforYavas = np.array([[sum([(Test_Yd_Sinif[i] == true_class) and (Test_Yd_hesaplanan_Sinif[i] == pred_class)
                                      for i in range(len(Test_Yd_Sinif))])
                                 for pred_class in range(0, sinifsayisinihai)]
                                for true_class in range(0, sinifsayisinihai)])
    accuracy_yavas = round(ConfusionMatrixforYavas[1][1] / np.sum(ConfusionMatrixforYavas, axis=1)[1], 3)
    # Calculate accuracy for "yavas" class.
    specificity_for_class_1 = round(tn1 / (tn1 + fp1), 3)
    specificity_for_class_0 = round(tn2 / (tn2 + fp2), 3)
    specificity=round((specificity_for_class_1 + specificity_for_class_0)/2, 3)
    balanced_accuracy = round(balanced_accuracy_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif), 3)
    recall = round(recall_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif,average="macro"), 3)
    precision = round(precision_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif,average="macro"), 3)
    fmeasure = round(f1_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif,average="macro"), 3)
    weighted_roc_auc_ovo = roc_auc_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif, multi_class="ovo", average="weighted")
    weighted_roc_auc_ovr = roc_auc_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif, multi_class="ovr", average="weighted")
    return accuracy, accuracy_yavas, recall, precision, fmeasure, confusionmatrix, weighted_roc_auc_ovo, weighted_roc_auc_ovr, specificity, balanced_accuracy,specificity_for_class_1,specificity_for_class_0

def accuracycheck(accuracy, accuracy_yavas):
    mevcutdurum=0
    if accuracy >= 0.80:
        if accuracy_yavas >=0.70:
            mevcutdurum=1
    else:
        mevcutdurum=0
    return mevcutdurum

# FLOW #
Training_Set, Random_Training_Index, Test_Set, Test_Set_Index, NumberofSample_TestSet = Generate_Training_Test_Set(Input)

# divide trainingset into two different sub-set
samplesize=round(len(Training_Set)/2)
Training_Set_SOM=Training_Set[:samplesize,:]
Training_Set_MLP=Training_Set[samplesize:,:]

# divide testset into n (pieces_testset) different sub-set
X=round(NumberofSample_TestSet/pieces_testset)  # number of sample after cute test set to 10 pieces
kucukx = 1
for i in range (0,pieces_testset):
    A=np.vstack((Test_Set[0, :], Test_Set[kucukx:kucukx + X, :]))
    TestSetS.append(A)
    kucukx = kucukx + X
TestSetS.append(Test_Set[kucukx:,:])

# train the model
W0_son, W2_son, W1_son, E_ani_egitim_deger, egitim_itr_sayisi, n2, n1 = mlp_training(Training_Set_MLP[:,1:-sinifsayisi],Training_Set_MLP[:,-sinifsayisi:])

for f in range (0,pieces_testset):
    print(f,". döngü:")
    nf1 = dt.datetime.now()
    featureselectioncount=0
    durum=0

    while durum==0:
        SelecetedFeatures_index, TrainSet_With_Selectedindex, dim_Som = Feature_Selection(Features_index_const, Training_Set_SOM)
        SelecetedFeatures_index, TestSet_With_Selectedindex, dim_Som = Feature_Selection(SelecetedFeatures_index, TestSetS[f])
        #print("TestSetS[f]",TestSetS[f])
        #print("TestSet_With_Selectedindex", TestSet_With_Selectedindex)
        if len(Features_index_const) == 0:
            w_test, Winners, EgitimSureSOM, som_sinifbilgisi, obek_etiketi, silinen_agliklar, neuronnumber_test = SOM_training(neuronnumber, TrainSet_With_Selectedindex, dim_Som)
        Test_Yd_SOM = SOM_test(neuronnumber_test, TestSet_With_Selectedindex[:,1:-sinifsayisi], w_test, obek_etiketi)


        Potential_Yavas=[]
        for k in range (0,len(Test_Yd_SOM)):
            if Test_Yd_SOM[k]==sinifsayisi-1:
                Potential_Yavas.append(TestSet_With_Selectedindex[k+1,0])
        Potentialindex = np.array(Potential_Yavas)
        #print("Potentialindex",Potentialindex)
        Seleceted_index = []
        Potential_Yavas_TestSet=[]
        if len(Potentialindex) == 0:
            print("Potentialindex BOS GELDİ!!!")
        else:
            for h in Potentialindex:
                index = np.argwhere(Test_Set[:, 0] == h)
                index = np.reshape(index, (1, len(index)))  # find the index of each randomly selected features
                Seleceted_index = np.append(Seleceted_index, index[0])
                Seleceted_index = Seleceted_index.astype(int)
            Potential_Yavas_TestSet = Test_Set[Seleceted_index, :]
            Test_Yd_hesaplanan_Sinif = mlp_test(Potential_Yavas_TestSet[:,1:-sinifsayisi], W0_son, W2_son, W1_son)
            Potential_Yavas_TestSet_Reel=[]
            for k in range(0, len(Potential_Yavas_TestSet)):
                for l in range(0, sinifsayisi):
                    if Potential_Yavas_TestSet[k][-sinifsayisi+l] == 1:
                        Potential_Yavas_TestSet_Reel.append(l)
            accuracy, accuracy_yavas, recall, precision, fmeasure, cm, weighted_roc_auc_ovo, weighted_roc_auc_ovr,specificity, balanced_accuracy,specificity_for_class_1,specificity_for_class_0 = performance(Potential_Yavas_TestSet_Reel,Test_Yd_hesaplanan_Sinif)
            print("Potential_Yavas_TestSet_Reel",Potential_Yavas_TestSet_Reel)
            print("Test_Yd_hesaplanan_Sinif",Test_Yd_hesaplanan_Sinif)
            print("cm",cm)
            durum = accuracycheck(accuracy, accuracy_yavas)

        if durum==0:
            if featureselectioncount<=featureselectionnumber:
                SelecetedFeatures_index = []
                featureselectioncount=featureselectioncount+1
        else:
            Features_index_const=SelecetedFeatures_index
            durum=1
        if featureselectioncount==10:
            break
        print("durum",durum)
        print("featureselectioncount",featureselectioncount)
    Potentialindex_ALL+=Potential_Yavas
    Potentialpredict_ALL+=Test_Yd_hesaplanan_Sinif
    Accuracy_List.append(accuracy)
    Accuracy_yavas_List.append(accuracy_yavas)
    Precision_List.append(precision)
    Recall_List.append(recall)
    F_Measure_List.append(fmeasure)
    roc_auc_ovo_List.append(weighted_roc_auc_ovo)
    roc_auc_ovr_List.append(weighted_roc_auc_ovr)
    Specificity_List.append(specificity)
    Balanced_accuracy_List.append(balanced_accuracy)
    specificity_for_class_1_List.append(specificity_for_class_1)
    specificity_for_class_0_List.append(specificity_for_class_0)
    nf2 = dt.datetime.now()
# FLOW #
#print("Potentialindex_ALL",Potentialindex_ALL,"len",len(Potentialindex_ALL))
#print("Potentialpredict_ALL",Potentialpredict_ALL,"len",len(Potentialpredict_ALL))
for i in range(1,numberofinput+1):
    if i in Test_Set[1:,0]:
        if i in Potentialindex_ALL:
            index=Potentialindex_ALL.index(i)
            predict_ALL.append(Potentialpredict_ALL[index])
        else:        predict_ALL.append(0)


print("Test_Set[1:,-1]",Test_Set[1:,-1])
print("predict_ALL",predict_ALL)
nS = dt.datetime.now()

print("")
print("PERFORMANS METRİKLERİ")
print("Accuracy: ", Accuracy_List)
print("Accuracy_Yavas: ", Accuracy_yavas_List)
print("Precision_Ort: ",Precision_List)
print("Recall_Ort: ",Recall_List)
print("F-Meause_Ort: ",F_Measure_List)
print("ROC_AUC_OVO: ", roc_auc_ovo_List)
print("ROC_AUC_OVR: ", roc_auc_ovr_List)
print("Specificity: ", Specificity_List)
print("Balanced_accuracy: ", Balanced_accuracy_List)
print("Specificity_for_class_1: ", specificity_for_class_1_List)
print("Specificity_for_class_0: ", specificity_for_class_0_List)
print(" ")
print("PERFORMANS METRİKLERİ - ORT")
print("Accuracy_Ort: ", np.mean(Accuracy_List))
print("Accuracy_Yavas_Ort: ", np.mean(Accuracy_yavas_List))
print("Precision_Ort: ",np.mean(Precision_List))
print("Recall_Ort: ", np.mean(Recall_List))
print("F-Meause_Ort: ", np.mean(F_Measure_List))
print("ROC_AUC_OVO_Ort: ", np.mean(roc_auc_ovo_List))
print("ROC_AUC_OVR_Ort: ", np.mean(roc_auc_ovr_List))
print("Specificity_Ort: ", np.mean(Specificity_List))
print("Balanced_accuracy_Ort: ", np.mean(Balanced_accuracy_List))
print("Specificity_for_class_1_Ort: ", np.mean(specificity_for_class_1_List))
print("Specificity_for_class_0_Ort: ", np.mean(specificity_for_class_0_List))

print("Egitim iterasyon sayısı", egitim_itr_sayisi)
print("Sure MLP Egitim:",n2-n1)
print("Sure Test", nf2-nf1)
print("Sure Toplam:",nS-n0)
