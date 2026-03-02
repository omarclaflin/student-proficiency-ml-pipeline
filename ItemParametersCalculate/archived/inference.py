import pandas as pd
import numpy as np
import base64
import datetime
import os

#Change working directory for the docker instance
# Get the absolute path to the script file
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
# Change the current working directory to the script directory
os.chdir(script_dir)
print('changed working directory to' + script_dir)


#some snowflake connector issues after update
try:
    #!pip install -force-reinstall pyarrow==10.0    
    import snowflake.connector
except:
    #!pip install snowflake-connector-python
    #!pip install -force-reinstall pyarrow==9.0
    #print('upgrading/checking pip')
    #os.system('pip install --upgrade pip')
    print('installing snowflake')
    os.system("pip install 'snowflake-connector-python'")
    #os.system("pip install 'snowflake-connector-python==2.7'")
    import snowflake.connector

# !pip install --upgrade numexpr

#ignore divide by zero warnings (when converging student deltas)
np.seterr(divide='ignore', invalid='ignore')

def getcode(fname='../credentials/mlAccountCode'):
    f=open(fname)
    s=f.read()
    f.close()
    return s.strip()

def getUsername(fname='../credentials/mlAccountUsername'):
    f=open(fname)
    s=f.read()
    f.close()
    return s.strip()
    

def query_for_skills():
    try:
        import snowflake.connector
    except:
        get_ipython().system('pip install snowflake-connector-python')
        import snowflake.connector


    import base64
    WAREHOUSE = 'COMPUTE_WH'
    USER = getUsername() #[REDACTED]
#     code = getcode()
#     PASSWORD = base64.b64decode(code).decode("utf-8")
    PASSWORD = getcode()
    ACCOUNT='[REDACTED]'
    DATABASE = '[REDACTED]'
    #print('usn: "', USER, '"')
    #print('pwd: "', PASSWORD, '"')

    ctx = snowflake.connector.connect(
        user=USER,
        password=PASSWORD,
        account=ACCOUNT, 
        database=DATABASE
    #     schema=SCHEMA
        )

    # Retrieving a Snowflake Query ID
    cur = ctx.cursor()
    cur.execute("USE warehouse COMPUTE_WH;")
    cur.execute("USE DATABASE [REDACTED];")
    #cur.execute("SELECT DISTINCT RL_TOP_LEVEL_SKILL_ID from [REDACTED].views.math_answers_fact")
    cur.execute(str("select DISTINCT(skill_or_subskill_id) as RL_TOP_LEVEL_SKILL_ID from "+
        "[REDACTED].prod.math_question_skills as skill_list WHERE removed='FALSE'"))

    print(cur.sfqid)

    # Get the results from a query.
    query_id=cur.sfqid
    cur.get_results_from_sfqid(query_id)
    results = cur.fetchall()
    cur.close()
    #print(f'{results}')
    if type(results[0])==tuple: #,type(skill_id[0])==str
        results = [item[0] for item in results]
    return results




#probably need to clean out s3 location

def downloadFromS3(bucket_name = '[REDACTED]', local_dir = '../AdaptiveMath'):
    #move from s3
    from boto3.session import Session
    import os

    if local_dir in os.listdir('.'):
        import shutil
        shutil.rmtree(local_dir)
        print(local_dir, ' removed.')
    os.mkdir(local_dir)

    session = Session()
    s3 = session.resource('s3')
    # s3_connection=boto.connect_s3()
    # s3_client = boto3.client('s3')
    bucket = s3.Bucket(bucket_name)
    # for file in os.listdir('AdaptiveMath'):
    # bucket=s3.Bucket(bucket_name)
    for s3_file in bucket.objects.all():
        print(s3_file)
        file_object = s3_file.key
        filename=str(file_object.split('/')[-1])
        bucket.download_file(file_object,local_dir+'/'+filename)


def downloadFileFromS3(bucket_name = '[REDACTED]', local_dir = '.', filename = 'ItemParameters.csv'):
    #move from s3
    from boto3.session import Session
    import os
    
    file_path=local_dir+'/'+filename

    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"The file {file_path} has been deleted.")
    else:
        print(f"The file {file_path} does not exist.")

    session = Session()
    s3 = session.resource('s3')
    # s3_connection=boto.connect_s3()
    # s3_client = boto3.client('s3')
    bucket = s3.Bucket(bucket_name)
    # for file in os.listdir('AdaptiveMath'):
    # bucket=s3.Bucket(bucket_name)
    for s3_file in bucket.objects.all():
        file_object = s3_file.key
        awsfilename=str(file_object.split('/')[-1])
        if awsfilename==filename:
            bucket.download_file(file_object,local_dir+'/'+awsfilename)
            print('downloading ', s3_file)
        

def read_filter_csv(filename, names, skill_id):
    import pandas as pd
    df = pd.read_csv(filename, names =names)
    df = df[df['rl_top_level_skill_id']== skill_id][['student_id','math_question_id','correctness','created_at']]
    return df

def loadAndFilterIntoDataframe(skill_id = '02a3bfaa479de311b77c005056801da1', limit=1000000, local_dir='../AdaptiveMath', 
                              parallel = True):
    #load and extract gzip csv files
    #need to dump into a user x question matrix (for each skill probably separately)
    import pandas as pd
    import os

    #header info so the file gets read correctly
#     names =['student_id', 'session_id', 'created_at', 'product', 'math_question_id',
#                                      'rl_top_level_skill_id', 'duration_seconds', 'correctness', 'attempts',
#                                     'answer']
    names =['student_id', 'session_id', 'created_at', 'math_question_id',
                                    'correctness', 'rl_top_level_skill_id']


    if parallel:
        files = os.listdir(local_dir)
        filelist = [file for file in files if file[-6:] == 'csv.gz']
        
        #test only a few
        #filelist=filelist[:10]
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-3)(delayed(read_filter_csv)(local_dir + '/'+file, 
                                names, skill_id) for file in filelist)
        df = pd.concat(results,axis=0,ignore_index=True)        
    if not parallel:
        #non parallel way to do it
        big_df_list=[]

        files = os.listdir(local_dir)
        filelist = [file for file in files if file[-6:] == 'csv.gz']

        for file in filelist:
            df = read_filter_csv(local_dir+'/'+file, names, skill_id)

            #df = df[df['rl_top_level_skill_id']== skill_id][['student_id','math_question_id','correctness','created_at']]
            big_df_list.append(df)

        df = pd.concat(big_df_list,axis=0,ignore_index=True)
        
    import time
    start = time.time()
    df = df.sort_values(by=['created_at'])
    print('done sorting, took ', time.time()-start, ' to sort.')
    
    return df[['student_id','math_question_id','correctness']].head(limit)

def returnTable(df):
    #df into table
    table = pd.pivot_table(df, values='correctness', index='student_id', columns='math_question_id',aggfunc='first')
    #table = table/100
    import numpy as np
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
    import numpy as np
    return c + (1-c)*np.exp(a*(theta-b))/(1+np.exp(a*(theta-b)))

def four_param_logistic(theta,a,b,c,d):
    #theta: student ability
    #b: difficulty parameter
    #a: discrimination parameter
    #c: guessing parameter
    #d: inattention parameter
    import numpy as np
    return c + (d-c)*np.exp(a*(theta-b))/(1+np.exp(a*(theta-b)))



def estimate_parameters_for_skill(table, thetas, PLOT_ON=True, FOUR_PL=True,
                                 bounds = None):
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
                print('modelling broke, skipping item ', table.columns[item_num]) 
                #this can occur because few responses and they're all correct for instance
    if FOUR_PL:
        return all_discriminability, all_difficulty, all_guessing, all_attention_errors
    else:
        return all_discriminability, all_difficulty, all_guessing


# all_discriminability, all_difficulty, all_guessing = estimate_parameters_for_skill(table)


def plot_item_with_model(thetas, item_num, table, popt, FOUR_PL=True):
    #input student thetas, responses for item
    #thetas: student thetas
    #item_num & table: response table and index to select
    #popt: model parameters
    #plots Item Characterisitic (probability) plot
    item=table.columns[item_num]
    item_series = table[[item]][table[item].notna()]
    item_thetas = thetas[table[item].notna()]
    
    import matplotlib.pyplot as plt
    plt.title('Item '+table.columns[item_num])
    plt.scatter(item_thetas,item_series[item])
    plt.plot(np.sort(item_thetas),
             model(np.sort(item_thetas),*popt))
    plt.show()
    print('a :', popt[0], ' b: ', popt[1], ' c: ',popt[2])
    if FOUR_PL:
        print('d: ', popt[3])    
        
def plot_information_curves(table,thetas,est_params, x_axis=None):
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
    import matplotlib.pyplot as plt
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
    import matplotlib.pyplot as plt

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
    outlier=int(len(solvedIRT_5.thetas)*(exclusion_percentage/100))
    indx = np.argsort(solvedIRT_5.thetas)[outlier:-1*outlier]
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
                print('modelling broke, skipping item ', table.columns[item_num]) 
        else:
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



def solve_IRT_for_matrix(table, all_thetas = None, iterations = 50, FOUR_PL=True, 
                         show_convergence=10, bounds=None):
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
                plt.show()
                
        #if theta range is already extreme, end estimation
        if max(all_thetas)-min(all_thetas)>6:
            #print('theta ranges: ',max(all_thetas)-min(all_thetas),max(all_thetas),min(all_thetas))
            break
                
                
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
            from sklearn.metrics import roc_curve
            fpri, tpri, thresholds = roc_curve(item_series.values, predicted)
            optimal_thresholdi = thresholds[np.argmax(tpri-fpri)]

            from sklearn.metrics import roc_auc_score
            #return roc auc score and threshold (and tpr and tnr for that threshold)
            auc_roc_all[item_num],optimal_threshold_all[item_num],tpr_all[item_num],tnr_all[item_num] = roc_auc_score(item_series.values, predicted), optimal_thresholdi, tpri[np.argmax(tpri-fpri)], 1-fpri[np.argmax(tpri-fpri)]
            #print('auc_roc,optimal_threshold,tpr,tnr ', auc_roc_all[item_num],optimal_threshold_all[item_num],tpr_all[item_num],tnr_all[item_num])
        except:
            print('skipped AUC ROC, opt thresh, tpr, tnr for item ', table.columns[item_num])
    
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

          
            
def submitSnowflakeQuery(query):
    try:
        import snowflake.connector
    except:
        get_ipython().system('pip install snowflake-connector-python')
        import snowflake.connector


    import base64
    WAREHOUSE = 'COMPUTE_WH'
    USER = getUsername()
    PASSWORD = getcode()
    ACCOUNT='[REDACTED]'
    DATABASE = '[REDACTED]'
    # SCHEMA = ''
    # Gets the version
    ctx = snowflake.connector.connect(
        user=USER,
        password=PASSWORD,
        account=ACCOUNT, 
        database=DATABASE
    #     schema=SCHEMA
        )
    cs = ctx.cursor()
    try:
        cs.execute("USE warehouse COMPUTE_WH;")
        #cs.execute("USE DATABASE [REDACTED];")        
        cs.execute(query)
        all_rows = cs.fetchall()
        print(all_rows)
        #one_row = cs.fetchone()
        #print(one_row[0])
    finally:
        cs.close()
    ctx.close()



def export_object_to_csv(solvedIRT, skill_id, filename='estimatedItemParameters.csv', version='1.0', no_csv_export=False):
    #CHANGE THIS SO IT UPDATES/APPENDS CSV FILE
    #CHANGE SO IT SHIFTS ALL BETAS TO POSITIVE?
    #inputs solved IRT object with all estimated parameters
    #exports a 10 field csv with 4 estimated parameters and 4 error scores for each question_id
    import numpy as np
    import datetime

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
        import os
    #     path = 'itemParametersBySkill'
    #     if not os.path.isdir(path):
    #         os.mkdir(path)
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


        

    

def grabAllDataFromSnowflake(stage='[REDACTED]', local_dir='../', 
                             exists=False, drop=True, backupS3=False, backup_old=False,
                             MIN_ITEMS=5, APIVERSION=6):
    #Build stage on snowflake & query for all data
    #exists: if True, doesn't requery but uses existing stage
    #drop: if True, drops/cleans up stage on snowflake

    #skip this if we don't want to requery and make a new stage
    if not exists:
        import time
        starttime = time.time()
        ctx = snowflake.connector.connect(
            user=USER,
            password=PASSWORD,
            account=ACCOUNT, 
            database=DATABASE
        #     schema=SCHEMA
            )
        cs = ctx.cursor()
        # try:
        print('Creating stage: '+stage+' on Snowflake and filling it up with data')
        
        cs.execute("USE warehouse COMPUTE_WH;")
        cs.execute("USE DATABASE [REDACTED];")
        try:
            cs.execute("drop stage "+stage)
            print('stage existed already. dropped.')
        except:
            print('stage did not exist. creating it.')
            
        cs.execute("CREATE STAGE "+stage)
        # cs.execute("copy into @[REDACTED]/result/data_ from (select * from [REDACTED].views.math_answers_fact) file_format=(compression='gzip');")
        # cs.execute("copy into @[REDACTED]/result/data_ from (select * from [REDACTED].views.math_answers_fact) file_format=(compression='gzip');")
        #This command has a new join to cross-list all skills for each question (that has multiple skill associations) by duplicating rows
        #Also checks to see if that skill-question association is up to date
              
        #Old query
#         subquery=("select answers.student_id, answers.session_id, "+
#            "answers.created_at, answers.math_question_id, answers.correctness, "+
#            "skill_list.skill_or_subskill_id as RL_TOP_LEVEL_SKILL_ID from "+
#            "[REDACTED].views.math_answers_fact as answers JOIN "+
#            "[REDACTED].prod.math_question_skills as skill_list "+
#            "ON answers.MATH_QUESTION_ID = skill_list.MATH_QUESTION_ID "+
#            "WHERE removed='FALSE' and answers.created_at >= dateadd(year, -1, current_date)"+
#            "ORDER BY answers.created_at")
            
        #ProgressAPI
        #names =['student_id', 'session_id', 'created_at', 'math_question_id',
        #                        'correctness', 'rl_top_level_skill_id']
        
        
        subquery=("SELECT student_id, token_id, created_at, question_id as math_question_id, correctness, skill_or_subskill_id "
            + "FROM ( select STUDENTID as student_id, metadata:ANSWERTOKEN as token_id, "
            + "OCCURREDAT as created_at, metadata:questionId as question_id, "
            + "metadata:itemCorrectness::NUMBER as correctness, "
            + "CASE WHEN rlsubSkillId IS NOT NULL THEN rlsubSkillId ELSE rlskillId END AS skill_or_subskill_id, "
            + "COUNT(*) OVER (PARTITION BY STUDENTID, skill_or_subskill_id) AS total_num, "
            + "ROW_NUMBER() OVER (PARTITION BY STUDENTID, skill_or_subskill_id ORDER BY created_at) AS trial_number "
            + "FROM progress.proficiency_events "
            + "WHERE version = 'progress@"+str(APIVERSION)+"' AND created_at >= dateadd(month, -3, current_date) "
            + "ORDER BY skill_or_subskill_id, STUDENTID, OCCURREDAT "
            + ") WHERE trial_number <=30 and total_num>="+str(MIN_ITEMS))
        
        
        query = str("copy into @"+stage+
           "/result/data_ from ("+subquery+
           ") file_format=(compression='gzip');")

        print('submitting query: '+ query)
        cs.execute(query)

        #datecutoff = '2019-08-09'
        #cs.execute("copy into @"+stage+"/result/data_ from (select answers.student_id, answers.session_id, answers.created_at, answers.math_question_id, answers.correctness, skill_list.skill_or_subskill_id as RL_TOP_LEVEL_SKILL_ID from [REDACTED].views.math_answers_fact as answers JOIN [REDACTED].prod.math_question_skills as skill_list ON answers.MATH_QUESTION_ID = skill_list.MATH_QUESTION_ID WHERE removed='FALSE' AND answers.created_at > "+ datecutoff +" SORT BY answers.created_at) file_format=(compression='gzip');")

        print('Completed query and stage copy: '+ str(time.time()-starttime) + ' seconds.')




    #pull down snowflake stage to local drive
    
    #make local dir
    import os
    if not os.path.exists(local_dir+'AdaptiveMath'):
        print('making new local dir')
        os.mkdir(local_dir+'AdaptiveMath')
    else:
        if backup_old:
            #rename old folder using todays date as archive
            print('old directory found, renaming and making new local dir')
            import time
            suffix = str(int(time.time()))
            os.rename(local_dir+'AdaptiveMath',local_dir+'AdaptiveMath'+suffix)
            #make new dir
            os.mkdir(local_dir+'AdaptiveMath')    
        else:
            #just delete files in it
            import os
            import glob

            files = glob.glob(local_dir+'AdaptiveMath/*.csv.gz')
            for f in files:
                os.remove(f)
            print('done deleting '+str(len(files))+' files in Adaptive Math directory.')
                
                
    ctx = snowflake.connector.connect(
        user=USER,
        password=PASSWORD,
        account=ACCOUNT, 
        database=DATABASE
    #     schema=SCHEMA
        )
    cs = ctx.cursor()


    import time
    starttime = time.time()
    print('Transfering from Snowflake to local')
    cs.execute("USE warehouse COMPUTE_WH;")
    cs.execute("USE DATABASE [REDACTED];")
    import os
    cur_dir = os.getcwd()
    cs.execute("GET @"+stage +" file://"+cur_dir+'/'+local_dir+"/AdaptiveMath/;")
    print('Completed transfer from '+stage + 'to '+cur_dir +'/'+local_dir+'/AdaptiveMath/'+ str(time.time()-starttime) + ' seconds.')



    #should remove stage from snowflake 
    if drop:
        import time
        starttime = time.time()
        ctx = snowflake.connector.connect(
            user=USER,
            password=PASSWORD,
            account=ACCOUNT, 
            database=DATABASE
        #     schema=SCHEMA
            )
        cs = ctx.cursor()
        print('Cleaning up stage from Snowflake')

        cs.execute("USE warehouse COMPUTE_WH;")
        cs.execute("USE DATABASE [REDACTED];")
        cs.execute("drop stage "+stage)


        print('Stage '+stage+' dropped on Snowflake [REDACTED]. '+ str(time.time()-starttime) + ' seconds.')


    #move to s3
    #probably need to clean out s3 location

    if backupS3:
        import boto3, os
        s3_client = boto3.client('s3')
        bucket = '[REDACTED]'
        print('Backing up local to S3')

        import datetime
        currentdate = datetime.datetime.today().strftime("%Y%m%d")

        for file in os.listdir(cur_dir+'/'+local_dir + 'AdaptiveMath'):
            if file[-6:]=='csv.gz':
                response=s3_client.upload_file(cur_dir+'/'+local_dir + 'AdaptiveMath/'+file,bucket,'AdaptiveMathData'+currentdate+'/'+file)
            else:
                print(file, ' skipped.')
        print('Transferred files from '+cur_dir+'/'+local_dir + 'AdaptiveMath/'+file+' to '+bucket+'AdaptiveMathData'+currentdate+'/'+file+'.')

        
def writetolog(comment,filename="logfile"):
    import datetime
    now = datetime.datetime.now()
    nowtime = now.strftime("%d/%m/%Y %H:%M:%S")
    #f = open(filename, "a")
    mode = 'a' if os.path.exists(filename) else 'w'
    f= open(filename, mode)
    f.write(nowtime+': '+comment+'\n')
    f.close()
        
    


#PARAMETERS
WAREHOUSE = 'COMPUTE_WH'
USER = getUsername() #[REDACTED]
PASSWORD = getcode()
ACCOUNT='[REDACTED]'
DATABASE = '[REDACTED]'

base_version = "3" 
#version info for the IP code, not the recalc
#recalculations (with same version) are just datetime addended (e.g. v2.01012024)                    

skill_cycle = 10
#cycle of file uploads, set low for testing, high for efficiency of I/O

#biggest skill prob 02a3bfaa479de311b77c005056801da1

# list_of_skills = ['02a3bfaa479de311b77c005056801da1']
list_of_skills = None

use_existing_data=False
#if we don't want to requery and redownload

#<>0, if we're resuming this process after it broke 
restart_index=0


date_id_tag = datetime.datetime.today().strftime("%Y%m%d")

filename = 'ItemParameterEstimates_dockerproduced'+date_id_tag+'.csv'

version = base_version +"."+ date_id_tag
tablename = "ItemParameterEstimates_dockerproduced" + date_id_tag
logfile = "logfile"+date_id_tag    


#clear out old file if exists (back it up)
original_file_path='./'+filename
# Check if the file exists
if not os.path.isfile(original_file_path):
    print(f"The file {original_file_path} does not exist.")
else:
    # Extract the directory, name, and extension of the original file
    directory, filename = os.path.split(original_file_path)
    name, extension = os.path.splitext(filename)

    # Create a datetime tag
    datetime_tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Create the new file name with '_bu' and the datetime tag
    new_filename = f"{name}_bu{datetime_tag}{extension}"

    # Create the full path for the new file
    new_file_path = os.path.join(directory, new_filename)

    # Rename the file
    os.rename(original_file_path, new_file_path)
    print(f"Renamed file to {new_file_path}")




#are we starting from scratch? or just updating some skills in old file 
if list_of_skills == None:
    #new run? don't fetch existing cvs file, and query/grab all skills
    print('no skills submitted, not loading an existing csv file.\n')
    writetolog('no skills submitted, not loading an existing csv file.', logfile)
    
    #get all skills
    list_of_skills = query_for_skills()
    los = [i[0] for i in list_of_skills]
    print('fetched ', str(len(los)), ' skills.\n')
    writetolog('fetched '+ str(len(los))+ ' skills.', logfile)
    
else:
    #old run? download existing csv(to append/replace rows)
    # and don't query/fetch skill list
    downloadFileFromS3(bucket_name = '[REDACTED]', local_dir = '.', 
                       filename = 'ItemParameterEstimates.csv')
    

los = list_of_skills[:]
print(str(len(list_of_skills)), ' skills.')
writetolog(str(len(list_of_skills))+ ' skills.', logfile)


print('version: ', version, ' datetime: ', date_id_tag, ' filename: ', filename, 
      ' tablename: ', tablename, " logfile: ", logfile)
writetolog('version: '+ version+ ' datetime: '+ date_id_tag+ ' filename: '+ filename+ ' tablename: '+ tablename+" logfile: "+logfile, logfile)


    
    


if not use_existing_data:
    import time
    start=time.time()

    print('grabbing data with megaquery on all skills.\n')
    writetolog('grabbing data with megaquery on all skills.', logfile)

    grabAllDataFromSnowflake(stage="[REDACTED]",exists=False,drop=True,backupS3=False, 
                             backup_old=True, MIN_ITEMS=5, APIVERSION=6)

    print('data grab completed. '+str(time.time()-start)+'seconds. \n')
    writetolog('data completed.' +str(time.time()-start)+ 'seconds. ', logfile)



#restart=3891
count=0 + restart_index

skills_calculated = 0
for skill_id in los[restart_index:]:
    print('skill #: ',count+1,' skill_id: ', skill_id)
    writetolog('skill #: '+str(count+1)+' skill_id: '+str(skill_id), logfile)


    
    try:
        #ETL for this skill
        import time
        start=time.time()
        print('attempting to load skill from local data: '+str(skill_id))
        writetolog('attempting to load skill from local data: '+str(skill_id), logfile)
        df = loadAndFilterIntoDataframe(skill_id = skill_id, limit=1000000)
        print('loading time: ', time.time()-start, ' for : ', len(df), np.shape(df))
        writetolog('loading time: '+ str(time.time()-start)+ ' for : '+ str(len(df))+ str(np.shape(df)), logfile)

        import numpy as np

        df['correctness'] = df['correctness']/100
        df['correctness'] = df['correctness'].apply(np.floor)
        print('unique values (for binarization check): ', df['correctness'].unique())

        #print('subnormalization...')
        #normalization from perfect scores (0.01 ->0.99)
        #df['correctness']=df['correctness']*0.98+0.01
        #print('unique values (for sub-normalization check): ', df['correctness'].unique())
        #NEED TO UN-SUBNORMALZIE FOR SDT to wrok...

        print('converting into table...')
        writetolog('converting into table...', logfile)
        table=returnTable(df)

        print('table size: ', np.shape(table))
        writetolog('table size: '+str(np.shape(table)), logfile)


        #solve IRT
        print('solving for item parameters...')
        writetolog('solving for item parameters...', logfile)        
        starttime = time.time()
        solvedIRT_7 = solve_IRT_for_matrix(table, all_thetas = None, iterations = 250, 
                                           FOUR_PL=True, show_convergence=0,
                                           bounds = ((1,-3,0,.5),(100,3,.5,1)))
        print('parameter calculation time: ', time.time()-starttime, ' seconds.')
        writetolog('parameter calculation time: '+ str(time.time()-starttime)+ ' seconds.', logfile)

        
        if np.isnan(solvedIRT_7.est_params[1]).sum() > 0:
            print('nans in est params: '+ str( np.isnan(solvedIRT_7.est_params[1]).sum())+ ' out of ', np.size(solvedIRT_7.est_params[1]))
            writetolog('nans in est params: '+ str( np.isnan(solvedIRT_7.est_params[1]).sum())+ ' out of ' +str(np.size(solvedIRT_7.est_params[1])), logfile)
        if np.isnan(solvedIRT_7.auc_roc).sum()>0:
            print('nans in auc_roc: '+ str( np.isnan(solvedIRT_7.auc_roc).sum())+ ' out of '+ str( np.size(solvedIRT_7.auc_roc)))
            writetolog('nans in auc_roc: '+ str( np.isnan(solvedIRT_7.auc_roc).sum())+ ' out of '+ str( np.size(solvedIRT_7.auc_roc)), logfile)
        if np.isnan(solvedIRT_7.optimal_threshold).sum()>0:
            writetolog('nans in optimal_threshold: '+ str(np.isnan(solvedIRT_7.optimal_threshold).sum())+ ' out of '+ str(np.size(solvedIRT_7.optimal_threshold)), logfile)
            print('nans in optimal_threshold: '+ str(np.isnan(solvedIRT_7.optimal_threshold).sum())+ ' out of '+ str(np.size(solvedIRT_7.optimal_threshold)))


        #write out csv to local directory -- change this for lambda or other resource to appropriate dir
        print('exporting to csv...')
        writetolog('exporting to csv...', logfile)        

        export_object_to_csv(solvedIRT_7, skill_id, filename, version)
        print('exported.')
        writetolog('exported.', logfile)        

        skills_calculated +=1

    except:
        print('skipping ',skill_id, ' either loading or estimating broke.')
        writetolog('skipping '+skill_id+ ' either loading or estimating broke.', logfile)

        
    if ((count/skill_cycle) == int(count/skill_cycle) or count ==len(los[restart_index:])-1) and (skills_calculated>0):
        print('exporting to snowflake...')
        writetolog('exporting to snowflake...', logfile)

        import boto3, os
        s3_client = boto3.client('s3')
        response=s3_client.upload_file(filename,'product-snowflake-testing',filename)

        #create snowflake table
        submitSnowflakeQuery("create or replace table product_testing."+tablename+" (question_id TEXT, "+
                             "skill_id TEXT, discriminability FLOAT, difficulty FLOAT, guessing FLOAT, inattention FLOAT, "+
                            "discriminability_error FLOAT, difficulty_error FLOAT, guessing_error FLOAT, "+
                             "inattention_error FLOAT, auc_roc FLOAT, optimal_threshold FLOAT, tpr FLOAT, tnr FLOAT, "+
                             "skill_optimal_threshold FLOAT, student_mean_accuracy FLOAT, " +
                             "sample_size INTEGER, date_created DATETIME, version STRING)")

        #adding AUC_ROC, optimal_threshold, TPR, TNR, sample_size



        #copy data into sf table
        submitSnowflakeQuery("copy into product_testing."+tablename+" from @public.product_testing_stage/"+filename)
        print('exported to snowflake')
        writetolog('exported to snowflake', logfile)
        
        print('attempting s3 upload')
        writetolog('attempting s3 upload', logfile)

        #upload current csv file to s3 bucket
        import boto3, os
        s3_client = boto3.client('s3')

        origin=filename[:]
        bucket_name = '[REDACTED]'
        destination = 'test'+origin[:]

        print('tyring to uploade file: ', filename, ' to ', bucket_name, ' as ', destination)
        s3_client.upload_file(origin,
                              bucket_name,
                              destination,
                             ExtraArgs={'ACL':'bucket-owner-full-control'})
        
        print('uploaded to s3')
        writetolog('uploaded to s3', logfile)

        print('attempting to upload logfile: ', logfile, ' to ', bucket_name, ' as ', destination)
        s3_client.upload_file(logfile,
                              bucket_name,
                              destination,
                             ExtraArgs={'ACL':'bucket-owner-full-control'})
        
        print('uploaded to s3')
        writetolog('uploaded to s3', logfile)
    count+=1

print('finished with ',str(count),' skills queried and calculated.')
writetolog('finished with '+str(count)+' skills queried and calculated.', logfile)

#finished, export to prod inbox
bucket_name='[REDACTED]'
destination = '/parameters/math-questions/ItemParameterEstimates'+str(date_id_tag)+'.csv'

print('exporting to prod. trying to upload file: ', filename, ' to ', bucket_name, ' as ', destination)
writetolog('exporting to prod. trying to upload file: '+ filename+ ' to '+ bucket_name+ ' as '+ destination, logfile)
s3_client.upload_file(filename,
                      bucket_name,
                      destination,
                     ExtraArgs={'ACL':'bucket-owner-full-control'})

print('uploaded to prod inbox: ', bucket_name,destination)
writetolog('uploaded to prod inbox: '+bucket_name+destination, logfile)


# #test with query -- not necessary for production
# submitSnowflakeQuery("select * from product_testing.itemParameters")

#skill_id, len(los[restart_index:]), count, [si for si in range(len(los)) if skill_id==los[si]]



