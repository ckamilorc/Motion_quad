
# importing libraries
""" List  functions to analyze, filter, process, etc the calcium data
    It defines the following functions:
    Accepted cells : gets only the accepted cells from the HDF5 file selected previusly using the caiman GUI
    Normalize to Zero: bring all the raw signals to zero
    Get conditions: gets all the conditions from the .mat file
    vel_acc : calculates velocity and acceleration from the quad.mat file
    filter_quad: Filters all the artefacts from the motion data
"""


import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, shapiro, ttest_rel, spearmanr



#%%
# Function for taking only the accepted cells from the visualization GUI.
def acceptedcells(cells, Accepted):
    """gets only the accepted cells"""
    sigraw = cells[Accepted, :]
    return sigraw

# Function to normalize the de-noized set of traces
def norm_zero_sigraw(sig, sigraw_n):
    """ Normalize the sig raw data to zero"""

    for x in range(len(sig)):
        min=np.amin(sig[x])
        sigraw_n[x]=sig[x]-min
    return sigraw_n


# Function get the onsets of each condition
def getcond(Conditions):
    """Gets the 4 conditions in a single arrays"""

    Cond1 = np.zeros(20)
    Cond2=np.zeros(20)
    Cond3=np.zeros(20)
    Cond4=np.zeros(20)
    j = 0
    for x in range(0,len(Conditions['Cond1']),2):
        Cond1[j]=Conditions['Cond1'][x]
        j += 1
    j = 0
    for x in range(0,len(Conditions['Cond2']),2):
        Cond2[j]=Conditions['Cond2'][x]
        j += 1
    j =0
    for x in range(0,len(Conditions['Cond3']),2):
        Cond3[j]=Conditions['Cond3'][x]
        j += 1
    j = 0
    for x in range(0,len(Conditions['Cond4']),2):
        Cond4[j]=Conditions['Cond4'][x]
        j += 1
    j = 0
    Cond1=Cond1.astype(int)
    Cond2=Cond2.astype(int)
    Cond3=Cond3.astype(int)
    Cond4=Cond4.astype(int)
    return Cond1, Cond2, Cond3, Cond4

#function to filter artefacts from the running wheel
def filter_quad(quad):
    """Replace the values greater then 360 with the precedent value"""

    newquad=np.zeros(len(quad))
    i=0
    for element in quad:
        if element > 360 :
            newquad[i]=quad[i+10]

            i+=1
        else:
            newquad[i]=element
            i+=1
    return newquad

# Function to calculate instantaneus speed and acceleration
def vel_acc(cumquad,siglen, frame):

    """ gets the celocity from the cummulative quadrature filtered file"""
    # Initialization
    velocity=np.zeros(len(cumquad))
    acceleration=np.zeros(len(cumquad))

    for x in range(len(cumquad)-1):
        velocity[x] = (cumquad[x+1]-cumquad[x])/(frame[x+1]-frame[x])
    velocity=np.absolute(velocity)
    for x in range(len(cumquad)-1):
        acceleration[x] = (velocity[x+1]-velocity[x])/(frame[x+1]-frame[x])
    acceleration=np.absolute(acceleration)
    # slicing the velocity an acceleration to the same lenght of sigraw
    velocity= velocity[0:siglen]
    acceleration=acceleration[0:siglen]
    return velocity, acceleration

# Function reading variables
def read_var(cnm):
    """
    Extract the Variables from the Calcium File:
        F_dff: set of DF/F normalized activity traces
        C: set of temporal traces (each row of C corresponds to a trace)
        A: set of spatial footprints. Each footprint is represented in a column of A, flattened with order = 'F'
        Accepted: subselection of accepted components
    """
    F_dff = cnm.estimates.F_dff
    C = cnm.estimates.C
    dims=cnm.estimates.dims
    A=cnm.estimates.A
    Accepted=cnm.estimates.accepted_list
    return F_dff, C, dims, A, Accepted

#determine number of bins
def freedman_diaconis(data, returnas="width"):

    """
    Use Freedman Diaconis rule to compute optimal histogram bin width.
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively.


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin.
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    iqr = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    n = data.size
    bw = (2 * iqr) / np.power(n, 1/3)

    if returnas == "width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    return result


# Function to fet predictors
def get_predictors(vec_velo, window, max_win, num_pred):

    ''' it returns the X predictors for
    the velocity values, using different frame windows.
    it also returns the square of the predictors and
    the root square'''

    Xpred=np.zeros([num_pred,len(vec_velo)-2*max_win])
    # For loop for rolling the velocity vector each window
    for n in range(len(window)):
        temp=np.roll(vec_velo,window[n])
        Xpred[n]=temp[max_win:len(temp)-max_win]
    # Transposing the Predictors for the GLM matrix multiplication
    Xpred=np.transpose(Xpred)
    # Getting the squere of the predictors
    xtemp_square=np.square(Xpred)
    # Getting the rootsquare of the predictors
    xtemp_sqrt=np.sqrt(Xpred)
    # Appending both arrays into tho the initial predictors
    xtemp21=np.append(Xpred,xtemp_square,axis=1)
    Xpred21=np.append(xtemp21,xtemp_sqrt,axis=1)
    return Xpred, Xpred21

# Function to delete the threshold after the GLM
def delete_threshold(resid_respon,sig):

    ''' It returns the residuals after deleting the
    threshold using the firts zero of the raw signal'''
    residu= resid_respon+np.mean(sig)
    for n in range(len(sig)):
        if 1 < sig[n] and sig[n] < 10:
            const=residu[n]
            break
    residu=residu-const
    return residu


def cond_resiudals(Cond1, Cond2, Cond3, Cond4, window):
    '''Moves the  condition vector n frames, subtracting
     them from each condition'''
    Cond1res=Cond1-15
    Cond2res=Cond2-15
    Cond3res=Cond3-15
    Cond4res=Cond4-15
    return Cond1res, Cond2res, Cond3res, Cond4res


#Function responsive cells

def responsive_cells(signal,Condition, win, basemean):
    '''It founds whether the cell is responsive or not using non parametric test, compare=ing the base line vs
     the post-stimulus during a preselected window. The function returns the following parameters:

     responsive: 0 for non responsive, 1 for responsive
     stats: statistics reported by the wicolxon test
     p: p-value reported by the wicolxon tes
     base: array with the average of all the baselines taken before the onset for each trial
     post1: array with the average of all the post stimulus taken after the onset for each trial
     post2: array with the average of all the post stimulus taken  one window after the onset
     post3: post stimulus taken after of all the post stimulus taken two windows after the onset
     cor: spearman correlation value between the baseline and post stimulus
     corval: p-value reported by the spearman correlation'''

    # variables initialization
    baseline=np.zeros((len(Condition),win)) # 20 is the number of stimulation for condition
    poststim1=np.zeros((len(Condition),win))
    poststim2=np.zeros((len(Condition),win))
    poststim3=np.zeros((len(Condition),win))
    base=np.zeros(len(Condition))
    post1=np.zeros(len(Condition))
    post2=np.zeros(len(Condition))
    post3=np.zeros(len(Condition))

    for x in range(len(Condition)):
        #Getting the mean of the baseline
        mean_base=np.zeros(win)
        for j in range (win):
            mean_base[j]=signal[Condition[x]-win+j]

        if basemean == True:
            mean_base=np.mean(mean_base)
        else:
            mean_base=0
        #Getting the baseline and poststim minus baseline mean
        for j in range (win):
            baseline[x][j]=(signal[Condition[x]-win+j]-mean_base)
            poststim1[x][j]=(signal[Condition[x]+j]-mean_base)
            poststim2[x][j]=(signal[Condition[x]+win+j]-mean_base)
            poststim3[x][j]=(signal[Condition[x]+(2*win)+j]-mean_base)
        # Getting the mean value for each array
        base[x]=np.mean(baseline[x])
        post1[x]=np.mean(poststim1[x])
        post2[x]=np.mean(poststim2[x])
        post3[x]=np.mean(poststim3[x])
    # Shapiro to test normality
    statb,p=shapiro(base)
    statp,p=shapiro(post1)
    #If normal distribution use t test else use non parametric wilcoxon
    if statb== 1 and statp ==1:
        stat,p=ttest_rel(base,post1)

    else:
        stat,p=wilcoxon(base,post1)

    # Finding if the cell is responsive or not
    alpha = 0.05
    if p < alpha and np.mean(base)<np.mean(post1):
        responsive=1
    else:
        responsive=0

    cor, corval=spearmanr(base,post1)

    return responsive, stats, p, base, post1, post2, post3, cor, corval



