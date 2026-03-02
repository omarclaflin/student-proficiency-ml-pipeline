import pandas as pd
import numpy as np
import base64
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import glob
import boto3, os
import datetime        
import time


#ignore divide by zero warnings (when converging student deltas)
np.seterr(divide='ignore', invalid='ignore')


def returnTable(df, roundValues=True):
    #df into table
    table = pd.pivot_table(df, values='correctness', index='student_id', columns='math_question_id',aggfunc='first')
    if np.nanmax(table.values.flatten())>1:
        print('large correctness values detected, normalizing table')
        table = table/100
    if roundValues:
        print('roundValues=True, discretizing values')
        table = table.apply(np.floor)    #they put 50, 25 as correctness sometimes for 2nd,3rd attmepted correct
    
    #print('unique vals (check for non-binary correctness): ',table.unique())
    # table['general_score']=table.mean(axis=1)
    # table['num_weights']=table.notna().sum(axis=1)
    return table        

def three_param_logistic(theta,a,b,c):
    #theta: student ability
    #b: difficulty parameter
    #a: discrimination parameter
    #c: guessing parameter
    #d: inattention parameter
    return c + (1-c)*np.exp(a*(theta-b))/(1+np.exp(a*(theta-b)))

def four_param_logistic(theta,a,b,c,d):
    #theta: student ability
    #b: difficulty parameter
    #a: discrimination parameter
    #c: guessing parameter
    #d: inattention parameter
    return c + (d-c)*np.exp(a*(theta-b))/(1+np.exp(a*(theta-b)))

def estimate_parameters_for_skill(table, thetas, PLOT_ON=True, FOUR_PL=True,
                                 bounds = None):
    #table: table which is all users X all items for that skill, pivoted from pandas
    #needs to include a 'general_score', and a 'num_weights'
    MINIMUM_DATA_POINTS = 30
    #FOUR_PL=False

    num_items = len(table.columns)
    
    all_discriminability = np.empty((num_items))
    all_difficulty = np.empty((num_items))
    all_guessing = np.empty((num_items))
    all_discriminability[:] = np.nan
    all_difficulty[:] = np.nan
    all_guessing[:] = np.nan
    model = three_param_logistic
    if bounds == None:
        bounds = ((1,0,0),(100,1,.5)) #fix alpha as positive only
    #p0 = [.5, .5, 0]
    
    if FOUR_PL:
        model = four_param_logistic
        all_attention_errors = np.empty((num_items))
        all_attention_errors[:] = np.nan
        if bounds == None:
            bounds = ((1,0,0,.5),(100,1,.5,1))
        #p0 = [.5, .5, 0, 1]


    #LATER CAN PARALLELIZE THIS TO ESTIMATE ALL ITEMS AT ONCE
    for item_num in range(num_items):
        item=table.columns[item_num]
        item_series = table[[item]][table[item].notna()]
        item_thetas = thetas[table[item].notna()]        
        if np.shape(item_series)[0]>MINIMUM_DATA_POINTS: #probably needs ~50 datapoints but we'll be conservative for now about rejection
            item_series = item_series
            try:
                try:
                    popt,pcov = curve_fit(model,item_thetas,item_series[item],
                                          bounds = bounds, method='trf')
                    #print('size: ', np.shape(item_series))
                    #print('lm:',popt,pcov)
                except:
                    popt,pcov = curve_fit(model,item_thetas,item_series[item],
                                          bounds = bounds, method='dogbox')
                    #print('size: ', np.shape(item_series))
                    #print('dogbox:',popt,pcov)

                if PLOT_ON:
                    import matplotlib.pyplot as plt
                    plt.title('Item '+table.columns[item_num])
                    plt.scatter(item_thetas,item_series[item])

                    plt.plot(np.sort(item_thetas),
                             model(np.sort(item_thetas),*popt))
                    plt.show()
                    print('a :', popt[0], ' b: ', popt[1], ' c: ',popt[2])
                    if FOUR_PL:
                        print('d: ', popt[3])
                    
                all_discriminability[item_num] = popt[0]
                all_difficulty[item_num] = popt[1]
                all_guessing[item_num] = popt[2]
                if FOUR_PL:
                    all_attention_errors[item_num]=popt[3]
            except:
                if False:
                    print('modelling broke, skipping item ', table.columns[item_num]) 
                    #this can occur because few responses and they're all correct for instance
    if FOUR_PL:
        return all_discriminability, all_difficulty, all_guessing, all_attention_errors
    else:
        return all_discriminability, all_difficulty, all_guessing


# all_discriminability, all_difficulty, all_guessing = estimate_parameters_for_skill(table)


def plot_item_with_model(model, thetas, item_num, table, popt, FOUR_PL=True):
    #input student thetas, responses for item
    #thetas: student thetas
    #item_num & table: response table and index to select
    #popt: model parameters
    #plots Item Characterisitic (probability) plot
    item=table.columns[item_num]
    item_series = table[[item]][table[item].notna()]
    item_thetas = thetas[table[item].notna()]
    
    plt.title('Item '+table.columns[item_num])
    plt.scatter(item_thetas,item_series[item])
    plt.plot(np.sort(item_thetas),
             model(np.sort(item_thetas),*popt))
    plt.show()
    print('a :', popt[0], ' b: ', popt[1], ' c: ',popt[2])
    if FOUR_PL:
        print('d: ', popt[3])    
        
def plot_information_curves(model, table,thetas,est_params, x_axis=None):
    total_curve=np.zeros(100-1)

    #Information Curves
    theta_range = np.linspace(min(thetas),max(thetas),100)
    for item_num in range(len(table.columns)):
        if ~np.isnan(est_params[0][item_num]):
            a,b,c,d = [i[item_num] for i in est_params]
            plt.plot(theta_range[:-1], np.diff(model(theta_range,a,b,c,d)))
            total_curve+=np.diff(model(theta_range,a,b,c,d))
    plt.title('Info curves for all items')
    if not x_axis==None:
        plt.xlim(x_axis)
    plt.show()

    aa,bb=np.histogram(thetas,100)

    plt.plot(theta_range[:-1],total_curve/sum(total_curve))
    plt.plot(bb[:-1],aa/sum(aa))
    if not x_axis==None:
        plt.xlim(x_axis)
    plt.title('Info curve for entire skill (blue) \n and distribution of theta (orange)')
    plt.show()        
    
    
def distributionsOfEstimatedItemParameters(solvedIRT, FOUR_PL=True):
    plt.hist(solvedIRT.est_params[0], bins = int(2*np.sqrt(len(solvedIRT.est_params[0]))))
    plt.title('discriminability distribution of all items')
    plt.show()


    plt.hist(solvedIRT.est_params[1], bins = int(2*np.sqrt(len(solvedIRT.est_params[1]))))
    plt.title('difficulty distribution of all items')
    plt.show()


    plt.hist(solvedIRT.est_params[2], bins = int(2*np.sqrt(len(solvedIRT.est_params[2]))))
    plt.title('guessing distribution of all items')
    plt.show()

    if FOUR_PL:
        plt.hist(solvedIRT.est_params[3], bins = int(2*np.sqrt(len(solvedIRT.est_params[3]))))
        plt.title('inattention distribution of all items')
        plt.show()

        
def plot_sample_parameter_convergence(solvedIRT, sample_of_items = 10,sample_of_students = 100):
    #plot history of estimates across iterations

    plt.plot(solvedIRT.discriminability_hx[:,np.random.choice(np.shape(solvedIRT.discriminability_hx)[1],sample_of_items)])
    plt.title('History of estimates of discriminability')
    plt.show()

    plt.plot(solvedIRT.difficulty_hx[:,np.random.choice(np.shape(solvedIRT.difficulty_hx)[1],sample_of_items)])
    plt.title('History of estimates of difficulty')
    plt.show()

    plt.plot(solvedIRT.guessing_hx[:,np.random.choice(np.shape(solvedIRT.guessing_hx)[1],sample_of_items)])
    plt.title('History of estimates of guessing')
    plt.show()

    if True:
        plt.plot(solvedIRT.attention_hx[:,np.random.choice(np.shape(solvedIRT.attention_hx)[1],sample_of_items)])
        plt.title('History of estimates of inattention')
        plt.show()


    plt.plot(solvedIRT.student_thetas_hx[:,np.random.choice(np.shape(solvedIRT.student_thetas_hx)[1],sample_of_students)])
    plt.title('History of estimates of thetas')
    plt.show()

def timeCourseOfParameterConvergence(solvedIRT, exclusion_percentage=5):
    #plot history of estimates across iterations
    #exclude first X% (exclusion_percentage ) of runs b/c they're often large numbers

    listOfParameters = ['discriminability', 'difficulty', 'guessing', 'attention']
    
    exclude = exclusion_percentage/100
    for param in listOfParameters:
        history = eval('solvedIRT.'+param+'_hx')
        err_history = eval('solvedIRT.'+param+'_error_hx')

        fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
        ax1.plot(np.diff(np.nanmean(history,1),1))
        ax1.set_title('Mean history of changes \n in estimate of '+param)

        ax2.plot(np.nanmean(err_history,1))
        ax2.set_title('Mean history of model \n error in estimate of '+param)

        ax3.plot(np.nanmean(err_history[int(np.shape(err_history)[0]*exclude):,:],1))
        ax3.set_title('Mean history of model \n error in estimate of '+param+' \n excluding first '+str(exclusion_percentage)+'% runs')
        plt.show()    

def correlationOfParametersByPerformance(solvedIRT, exclusion_percentage = 10):
    plt.title("student correct % by estimated student ability (theta)")
    plt.scatter(solvedIRT.thetas, solvedIRT.item_correct)
    plt.show()
    print('Correlation: ', np.corrcoef(solvedIRT.thetas, solvedIRT.item_correct)[0][1])

    #X% outlier removal
    outlier=int(len(solvedIRT.thetas)*(exclusion_percentage/100))
    indx = np.argsort(solvedIRT.thetas)[outlier:-1*outlier]
    plt.title("Student Correct % by \n Estimated Student Ability (theta) \n "+str(exclusion_percentage)+"% OUTLIER removed")
    plt.scatter(solvedIRT.thetas[indx], solvedIRT.item_correct.values[indx])
    plt.show()
    print('Correlation: ', np.corrcoef(solvedIRT.thetas[indx], solvedIRT.item_correct.values[indx])[0][1])
    
    plt.title("item % correct by estimated item difficulty (beta) ")
    plt.scatter(solvedIRT.est_params[1], solvedIRT.student_correct)
    plt.show()
    indx = ~np.isnan(solvedIRT.est_params[1])
    print('Correlation: ', np.corrcoef(solvedIRT.est_params[1][indx], solvedIRT.student_correct[indx])[0][1])

    plt.title("item % correct by estimated item information (alpha) ")
    plt.scatter(solvedIRT.est_params[0], solvedIRT.student_correct)
    plt.show()
    indx = ~np.isnan(solvedIRT.est_params[0])
    print('Correlation: ', np.corrcoef(solvedIRT.est_params[0][indx], solvedIRT.student_correct[indx])[0][1])


def compareRuns(A,B):
    #inputs two solvedIRT objects; outputs correlation between them
    At = A.thetas
    Bt = B.thetas

    plt.hist(A.thetas,bins=100)
    plt.hist(B.thetas,bins=100)
    plt.title('distribution of thetas')
    plt.show()
    print('theta parameter correl: ', np.corrcoef(At[~np.isnan(At)],Bt[~np.isnan(At)])[0][1])

    Ae = A.est_params[0]
    Be = B.est_params[0]
    print('discriminability parameter correl: ', np.corrcoef(Ae[~np.isnan(Ae)],Be[~np.isnan(Ae)])[0][1])

    Ae = A.est_params[1]
    Be = B.est_params[1]
    print('difficulty parameter correl: ',np.corrcoef(Ae[~np.isnan(Ae)],Be[~np.isnan(Ae)])[0][1])

    Ae = A.est_params[2]
    Be = B.est_params[2]
    print('guessing parameter correl: ',np.corrcoef(Ae[~np.isnan(Ae)],Be[~np.isnan(Ae)])[0][1])

    Ae = A.est_params[3]
    Be = B.est_params[3]
    print('atttention parameter correl: ',np.corrcoef(Ae[~np.isnan(Ae)],Be[~np.isnan(Ae)])[0][1])

    Ae = A.auc_roc
    Be = B.auc_roc
    print('auc_roc parameter correl: ',np.corrcoef(Ae[~np.isnan(Ae)],Be[~np.isnan(Ae)])[0][1])

    Ae = A.optimal_threshold
    Be = B.optimal_threshold
    print('optimal_threshold parameter correl: ',np.corrcoef(Ae[~np.isnan(Ae)],Be[~np.isnan(Ae)])[0][1])
    
    Ae = A.tpr
    Be = B.tpr
    print('tpr parameter correl: ',np.corrcoef(Ae[~np.isnan(Ae)],Be[~np.isnan(Ae)])[0][1])

    Ae = A.tnr
    Be = B.tnr
    print('tnr parameter correl: ',np.corrcoef(Ae[~np.isnan(Ae)],Be[~np.isnan(Ae)])[0][1])
    

def parallel_estimate_parameters_for_skill(table, thetas, PLOT_ON=True, FOUR_PL=True, est_kernel='trf',
                                          parallel=True, bounds = None):
    #table: table which is all users X all items for that skill, pivoted from pandas
    #needs to include a 'general_score', and a 'num_weights'
    MINIMUM_DATA_POINTS = 30
    #FOUR_PL=False
    from scipy.optimize import curve_fit
    import numpy as np
    
    num_items = len(table.columns)
    
    all_discriminability = np.empty((num_items))
    all_difficulty = np.empty((num_items))
    all_guessing = np.empty((num_items))
    all_errors =  np.empty((num_items))
    all_discriminability[:] = np.nan
    all_difficulty[:] = np.nan
    all_guessing[:] = np.nan
    all_errors =  np.nan
    model = three_param_logistic
    if bounds==None:
        bounds = ((1,-30,0),(100,30,.5)) #fix alpha as positive only
    #p0 = [.5, .5, 0]
    
    if FOUR_PL:
        model = four_param_logistic
        all_attention_errors = np.empty((num_items))
        all_attention_errors[:] = np.nan
        if bounds == None:
            bounds = ((1,-30,0.001,.5),(100,30,.5,.999))
        #p0 = [.5, .5, 0, 1]


    #LATER CAN PARALLELIZE THIS TO ESTIMATE ALL ITEMS AT ONCE
    
#     for item_num in range(num_items):
    def process(item_num):
        item=table.columns[item_num]
        item_series = table[[item]][table[item].notna()]
        item_thetas = thetas[table[item].notna()]        
        if np.shape(item_series)[0]>MINIMUM_DATA_POINTS: #probably needs ~50 datapoints but we'll be conservative for now about rejection
            #if all are correct, or incorrect
            item_series = item_series
            try:
                try:
                    popt,pcov = curve_fit(model,item_thetas,item_series[item],
                                          bounds = bounds, method='trf')
                except:
                    popt,pcov = curve_fit(model,item_thetas,item_series[item],
                                          bounds = bounds, method='dogbox')

                if PLOT_ON:
                    import matplotlib.pyplot as plt
                    plt.title('Item '+table.columns[item_num])
                    plt.scatter(item_thetas,item_series[item])

                    plt.plot(np.sort(item_thetas),
                             model(np.sort(item_thetas),*popt))
                    plt.show()
                    print('a :', popt[0], ' b: ', popt[1], ' c: ',popt[2])
                    if FOUR_PL:
                        print('d: ', popt[3])

                if FOUR_PL:
                    #returns the coefficients, and the ERRORS (est cov matrix)
                    return popt[0], popt[1], popt[2], popt[3], np.sqrt(np.diag(pcov))
        #             return all_discriminability, all_difficulty, all_guessing, all_attention_errors
                else:
                    return popt[0], popt[1], popt[2], np.sqrt(np.diag(pcov))
        #             return all_discriminability, all_difficulty, all_guessing
            except:
                if False:
                    print('modelling broke, skipping item ', table.columns[item_num]) 
        else:
            if False:
                print('modelling skipped, skipping item ', table.columns[item_num]) 

        #this can occur because few responses and they're all correct for instance
        if FOUR_PL:
            return np.nan, np.nan, np.nan, np.nan, np.asarray([np.nan, np.nan, np.nan, np.nan])
        else:
            return np.nan, np.nan, np.nan, np.asarray([np.nan, np.nan, np.nan])

   
    from joblib import Parallel, delayed
    if parallel:
        results = Parallel(n_jobs=-2)(delayed(process)(i) for i in range(num_items))
    else:
        results = Parallel(n_jobs=1, backend='multiprocessing')(delayed(process)(i) for i in range(num_items))
        
    
    all_discriminability = np.asarray([results[i][0] for i in range(len(results))])
    all_difficulty = np.asarray([results[i][1] for i in range(len(results))])
    all_guessing = np.asarray([results[i][2] for i in range(len(results))])
    
    #stores errors for all parameters (3 or 4, dep on model)
    all_errors = np.asarray([results[i][4] for i in range(len(results))])
    if FOUR_PL:
        all_attention_errors=np.asarray([results[i][3] for i in range(len(results))])
                
    if FOUR_PL:
#             return popt[0], popt[1], popt[2], popt[3]
        return all_discriminability, all_difficulty, all_guessing, all_attention_errors, all_errors
    else:
#             return popt[0], popt[1], popt[2]
        return all_discriminability, all_difficulty, all_guessing, all_errors


# %%capture output


import numpy as np

#this is to estimate probability of student getting item correct
def prob_est(all_thetas, model,all_params):
    return np.asarray([model(all_thetas,*params) for params in all_params]).T

def update_thetas(all_thetas, FOUR_PL, all_est_params, table):
    #thetas, model, item parameters, table of responses
    #inputs: current item parameters (4 if 4PL, etc)
    #table: of all correctness of students x items
    #all probs: all current probabilities for correctness each student and item

    if FOUR_PL:
        model=four_param_logistic
        all_params = zip(all_est_params[0],all_est_params[1],all_est_params[2],all_est_params[3])
    else:
        #set model and current params
        model=three_param_logistic
        all_params = zip(all_est_params[0],all_est_params[1],all_est_params[2])

    all_probs=prob_est(all_thetas,model,all_params) #returns #items by #users
    all_delta_thetas = np.nansum(all_est_params[0]*(table[table.columns]-all_probs),1)/np.nansum(np.power(all_est_params[0],2)*all_probs*(1-all_probs),1)
    
    #MLE denominator
    all_est_delta_confidence = 1/np.nansum(np.power(all_est_params[0],2)*all_probs*(1-all_probs),1)
    #all_est_delta_confidence = 1/np.nansum(np.power(all_est_params[0],2)*all_probs*(1-all_probs),1)
    #all_est_delta_confidence = 1/(np.power(all_est_params[0],2)*np.nanmean(all_probs,1)*np.nanmean((1-all_probs),1))
    #print('all_thetas: ',all_est_delta_confidence)

    #if ((np.nansum(all_est_params[0],1)==0) or (np.nansum(all_probs,1)==0) or (np.nansum(1-all_probs,1)==0)):
    #print('aep,ap,1-ap: ',all_est_params[0])
    #print('aep,ap,1-ap: ',all_probs)
    #print('aep,ap,1-ap: ', 1-all_probs)
        
    return all_delta_thetas, all_est_delta_confidence

def custom_roc_curve(y_true, y_pred):
    y_true= np.array(y_true)
    y_pred=np.array(y_pred)
    
    # Sort predictions and corresponding true values
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    # Calculate TPR and FPR
    tpr = np.cumsum(y_true_sorted) / np.sum(y_true)
    fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true)
    

    # Remove duplicate points (auc roc needs monotonic datapoints)
    # Remove duplicate points to ensure strictly increasing FPR
    # Check for monoticity in FPR and TPR

    # Combine FPR and TPR
    points = np.column_stack((fpr, tpr, y_pred_sorted))
    
    # Remove duplicates while preserving order
    _, unique_indices = np.unique(points[:, :2], axis=0, return_index=True)
    unique_points = points[np.sort(unique_indices)]
    
    # Add (0,0) and (1,1) points if they're not already present
    if not np.any(np.all(unique_points[:, :2] == [0, 0], axis=1)):
        unique_points = np.vstack(([0, 0, np.inf], unique_points))
    if not np.any(np.all(unique_points[:, :2] == [1, 1], axis=1)):
        unique_points = np.vstack((unique_points, [1, 1, -np.inf]))
    
    # Sort points by FPR, then by TPR
    unique_points = unique_points[np.lexsort((unique_points[:, 1], unique_points[:, 0]))]
    
    fpr, tpr, thresholds = unique_points[:, 0], unique_points[:, 1], unique_points[:, 2]
    
    
    return fpr, tpr, thresholds


def solve_IRT_for_matrix(table, all_thetas = None, iterations = 50, FOUR_PL=True, 
                         show_convergence=10, show_discriminability=0, 
                         bounds=None, verbose=False):
    #initialize from scratch
    #history of student ability estimates (& changes over estimation)
    all_student_thetas_hx=np.full((iterations,np.shape(table)[0]), np.nan)
    all_student_delta_thetas_hx=np.full((iterations,np.shape(table)[0]), np.nan)
    #history of item parameter estimates
    all_discriminability_hx=np.full((iterations,len(table.columns)), np.nan)
    all_difficulty_hx=np.full((iterations,len(table.columns)), np.nan)
    all_guessing_hx=np.full((iterations,len(table.columns)), np.nan)
    all_attention_hx=np.full((iterations,len(table.columns)), np.nan)

    #iterations x #items
    all_item_confidence_hx=np.full((iterations,len(table.columns)), np.nan)
    all_item_power_hx=np.full((iterations,len(table.columns)), np.nan)

    #iterations x #students
    all_theta_confidence_hx=np.full((iterations,np.shape(table)[0]), np.nan)
    all_theta_power_hx=np.full((iterations,np.shape(table)[0]), np.nan)

    #capture item paraemter error (iterations x items)
    all_discriminability_error_hx=np.full((iterations,len(table.columns)), np.nan)
    all_difficulty_error_hx=np.full((iterations,len(table.columns)), np.nan)
    all_guessing_error_hx=np.full((iterations,len(table.columns)), np.nan)
    all_attention_error_hx=np.full((iterations,len(table.columns)), np.nan)
    
    #if first run, then assume thetas unknown
    if all_thetas == None:
        all_thetas = table.mean(axis=1) #just create a thetas df
        #replace with random value for initialization
        all_thetas[all_thetas.index] = -.05 + .1*np.random.random((len(all_thetas.index)))


    for iter_num in range(iterations):
        if show_convergence>0:
            print("iteration #", iter_num)

        #if first or last, plot
        if iter_num==0 or iter_num==iterations-1:
            #run estimate function
            #all_discriminability, all_difficulty, all_guessing, all_attention
            #est_params = estimate_parameters_for_skill(table, True,FOUR_PL)
            all_est_params = parallel_estimate_parameters_for_skill(table, all_thetas, 
                                                                    False,FOUR_PL,bounds=bounds)
        else:
            #est_params = estimate_parameters_for_skill(table, False,FOUR_PL)
            all_est_params = parallel_estimate_parameters_for_skill(table, all_thetas, 
                                                                    False,FOUR_PL,bounds=bounds)

        #store history to examine later
        all_discriminability_hx[iter_num,:]=all_est_params[0]
        all_difficulty_hx[iter_num,:]=all_est_params[1]
        all_guessing_hx[iter_num,:] =all_est_params[2]

        #store error history
        all_discriminability_error_hx[iter_num,:]=all_est_params[-1][:,0]
        all_difficulty_error_hx[iter_num,:]=all_est_params[-1][:,1]
        all_guessing_error_hx[iter_num,:] =all_est_params[-1][:,2]

        if FOUR_PL:
            all_attention_hx[iter_num,:]= all_est_params[3]
            #store error history
            all_attention_error_hx[iter_num,:]= all_est_params[-1][:,3]

        all_student_thetas_hx[iter_num,:] = all_thetas

        #update table with new student thetas
        #new theta = old_theta + sum[(a * (response - prob) )]/sum[ a**2 * prob * (1-prob) ]
        # prob = probability of correct response to item i under current ICC model at currently estimated theta (old_theta)
        #delta_thetas = np.sum(all_discriminability*(table[table.columns[:-2]]-all_probs),1)/np.sum(all_discriminability**2*all_probs*(1-all_probs),1)
        #all_delta_thetas = np.nansum(all_est_params[0]*(table[table.columns]-all_probs),1)/np.nansum(np.power(all_est_params[0],2)*all_probs*(1-all_probs),1)

        all_delta_thetas, all_est_delta_confidence = update_thetas(all_thetas, FOUR_PL, all_est_params,table)        
        
        #if updated theta range is extreme, end estimation
        if max(np.nansum([all_thetas, all_delta_thetas],0))-min(np.nansum([all_thetas, all_delta_thetas],0))>6:
            print('convergence stopped, cycle: ',iter_num)
            #print('theta ranges: ',max(all_thetas)-min(all_thetas),max(all_thetas),min(all_thetas))
            break
        else:
            #update: theta + delta_theta
            all_thetas = np.nansum([all_thetas, all_delta_thetas],0)

        #number of items in estimate for this student's theta
        all_item_power = np.sum(table[table.columns].notna(),1)
        all_student_delta_thetas_hx[iter_num,:] = all_delta_thetas
        all_theta_confidence_hx[iter_num,:] =all_est_delta_confidence
        all_theta_power_hx[iter_num,:] =all_item_power

        #number of students in estimate for this items's params
        all_student_power = np.sum(table[table.columns].notna(),0)

        all_item_power_hx[iter_num,:] =all_student_power

        #if alphas are mostly negative, invert entire theta/beta spectrum
        if sum(all_est_params[0]<0) > sum(all_est_params[0]>=0):
            print('inverting')
            all_thetas*=-1

        if show_convergence > 0:
            show_convergence = int(show_convergence)
            if iter_num%10==0:
                import matplotlib.pyplot as plt
                plt.hist(all_delta_thetas, bins =100)
                plt.title('Delta thetas for all students for run #'+str(iter_num))
                plt.show()
                
        if show_discriminability > 0:
            show_discriminability = int(show_discriminability)
            if iter_num%10==0:
                import matplotlib.pyplot as plt
                plt.hist(all_est_params[0], bins =100)
                plt.title('Discriminability for all items for run #'+str(iter_num))
                plt.show()
                
                
    #after estimation is complete, calculate SDT accuracy/error
    auc_roc_all = np.full((len(table.columns)), np.nan)
    optimal_threshold_all = np.full((len(table.columns)), np.nan)
    tpr_all = np.full((len(table.columns)), np.nan)
    tnr_all = np.full((len(table.columns)), np.nan)
    
    for item_num in range(len(table.columns)):
        try:
            item=table.columns[item_num]
            item_series = table[[item]][table[item].notna()]
            item_thetas = all_thetas[table[item].notna()]
            model_params = [all_est_params[i][item_num] for i in range(4)]

            #print(np.shape(item_thetas))
            #print(np.shape(model_params), model_params)


            #def estimate_error(performances, thetas, model_params):
            predicted = four_param_logistic(item_thetas,*model_params)

            #new logic to handle non-binary differently with custom func
            if np.array_equal(item_series.values, item_series.values.astype(bool).astype(int)):
                #old method for discrete variables
                from sklearn.metrics import roc_curve
                fpri, tpri, thresholds = roc_curve(item_series.values, predicted)

                from sklearn.metrics import roc_auc_score
                #return roc auc score and threshold (and tpr and tnr for that threshold)
                auc_roc_all[item_num] = roc_auc_score(item_series.values, predicted)            
            else:
                if verbose:
                    print('continuous values detected, using custom ROC functions for ', table.columns[item_num])
                    
                #new functions for non-discrete performance data
                fpri, tpri, thresholds = custom_roc_curve(item_series.values, predicted)

                from sklearn.metrics import auc
                auc_roc_all[item_num] = auc(fpri, tpri)


            optimal_thresholdi = thresholds[np.argmax(tpri-fpri)]

            #check for infinities, put default value in instead
            if np.isinf(optimal_thresholdi) or np.isnan(optimal_thresholdi):
                print('ot is nan or inf: ',optimal_thresholdi)
                optimal_thresholdi = 0.5


            optimal_threshold_all[item_num],tpr_all[item_num],tnr_all[item_num] = optimal_thresholdi, tpri[np.argmax(tpri-fpri)], 1-fpri[np.argmax(tpri-fpri)]

            #print('auc_roc,optimal_threshold,tpr,tnr ', auc_roc_all[item_num],optimal_threshold_all[item_num],tpr_all[item_num],tnr_all[item_num])

        except:
            if verbose:
                print('skipped AUC ROC, opt thresh, tpr, tnr for item ', table.columns[item_num])
            #print('thresholds', thresholds)
            #print('itemseries.values',item_series.values)
            #print('predicted', predicted)
    
    class IRTResults(object):
        question_ids = table.columns
        thetas=all_thetas
        est_params = all_est_params
        delta_thetas=all_delta_thetas
        
        item_power = all_item_power #num items for this theta
        student_power = all_student_power #num students in each item
        student_correct = np.nansum(table,0)/all_student_power #% correct across all students
        item_correct = np.nansum(table,1)/all_item_power #% correct across all items
        est_delta_confidence=all_est_delta_confidence #factors in alpha and prob distribution
        
        delta_thetas_hx = all_student_delta_thetas_hx
        student_thetas_hx = all_student_thetas_hx 
        discriminability_hx = all_discriminability_hx
        difficulty_hx=all_difficulty_hx
        guessing_hx=all_guessing_hx
        attention_hx=all_attention_hx
        item_confidence_hx=all_item_confidence_hx
        item_power_hx=all_item_power_hx

        #iterations x #students
        theta_confidence_hx=all_theta_confidence_hx
        theta_power_hx=all_theta_power_hx

        #capture item paraemter error (iterations x items)
        discriminability_error_hx=all_discriminability_error_hx
        difficulty_error_hx=all_difficulty_error_hx
        guessing_error_hx=all_guessing_error_hx
        attention_error_hx=all_attention_error_hx
            
        auc_roc,optimal_threshold,tpr,tnr = auc_roc_all,optimal_threshold_all,tpr_all,tnr_all
        sample_size = student_power


    return IRTResults

          


def export_object_to_csv(solvedIRT, skill_id, filename='estimatedItemParameters.csv', version='1.0', no_csv_export=False):
    #CHANGE THIS SO IT UPDATES/APPENDS CSV FILE
    #CHANGE SO IT SHIFTS ALL BETAS TO POSITIVE?
    #inputs solved IRT object with all estimated parameters
    #exports a 10 field csv with 4 estimated parameters and 4 error scores for each question_id

    qid = solvedIRT.question_ids
    alpha,beta,gamma,epsilon = [solvedIRT.est_params[i] for i in range(4)]
    #center beta
    beta = beta + ((min(beta) - max(beta))/2) - min(beta)
    #could scale beta but then need to scale alpha -- skip for now, just interpretability for laypeople
    alpha_c,beta_c,gamma_c,epsilon_c = [np.asarray(solvedIRT.est_params[-1])[:,i] for i in range(4)]

    currentdate = datetime.datetime.today().strftime("%Y-%m-%d")
    
    auc_roc,optimal_threshold,tpr,tnr,sample_size = solvedIRT.auc_roc,solvedIRT.optimal_threshold,solvedIRT.tpr,solvedIRT.tnr,solvedIRT.sample_size

    skill_optimal_threshold = np.mean(solvedIRT.optimal_threshold) 

    student_correct=solvedIRT.student_correct
    
    export_df = pd.DataFrame({'question_id': qid,
                       'skill_id': skill_id,
                       'discriminability': alpha,
                       'difficulty': beta,
                       'guessing': gamma,
                       'inattention': epsilon,
                       'discriminability_error': alpha_c,
                       'difficulty_error': beta_c,
                       'guessing_error': gamma_c,
                       'inattention_error': epsilon_c,
                       'auc_roc': auc_roc,
                       'optimal_threshold': optimal_threshold,
                       'tpr': tpr,
                       'tnr': tnr,
                       'skill_optimal_threshold': skill_optimal_threshold, 
                       'student_mean_accuracy': student_correct, 
                       'sample_size': sample_size,
                       'date_created': currentdate,
                       'version': version})

    
    if no_csv_export:
        return export_df
    else:
        #make directory, store csv in that directory
        if not os.path.isfile(filename):
            export_df.to_csv(filename, index=False)
        else:
            #if it does exist, read in old file and overwrite existing skill_id/question_id lines
            
            # Read the existing CSV file into a DataFrame
            old_df = pd.read_csv(filename)

            # Concatenate the new DataFrame with the old one
            combined_df = pd.concat([export_df, old_df])

            # Drop duplicate rows based on 'question_id' and 'skill_id', keep the first occurrence
            final_df = combined_df.drop_duplicates(subset=['question_id', 'skill_id'], keep='first')

            if len(final_df) != len(combined_df):
                print('len(old), :', str(len(old_df)), 'len(final): ', str(len(final_df)), ' len(combined): ', str(len(combined_df)))
                print('overwriting ',str(len(combined_df)-len(final_df)),' old lines found in existing csv file...')
            # Write the result back to the CSV file
            final_df.to_csv(filename, index=False)
            
            #old method of just updating, assumed skill_id/question_id weren't in there
            #export_df.to_csv(filename, mode='a', header=False, index=False)            


        

    

        
def writetolog(comment,filename="logfile"):
    now = datetime.datetime.now()
    nowtime = now.strftime("%d/%m/%Y %H:%M:%S")
    f = open(filename, "a")
    f.write(nowtime+': '+comment+'\n')
    f.close()
        
if __name__ == "__main__":
    print("This module contains functions for IRT and SDT estimation of tabular student performance data.")
    