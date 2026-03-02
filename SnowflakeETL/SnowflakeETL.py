# try:
#     import snowflake.connector
# except:
#     !pip install snowflake-connector-python
#     import snowflake.connector


import sys
import subprocess

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    
try:
    import snowflake
except:
    install_package("snowflake")

try:
    import snowflake.connector
except:
    install_package("snowflake-connector-python")    
    
import snowflake.connector    
import snowflake
import time

import os

def getcode(fname='mlAccountCode'):
    credentials_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'credentials', fname)
    with open(credentials_path) as f:
        return f.read().strip()

def getUsername(fname='mlAccountUsername'):
    credentials_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'credentials', fname)
    with open(credentials_path) as f:
        return f.read().strip()
    
# def getcode(fname='mlAccountCode'):
#     f=open('../credentials/'+fname)
#     s=f.read()
#     f.close()
#     return s.strip()

# def getUsername(fname='mlAccountUsername'):
#     f=open('../credentials/'+fname)
#     s=f.read()
#     f.close()
#     return s.strip()

def grabAllDataFromSnowflake(stage='[REDACTED]', 
                             datastore="StudentProficiencyData",
                             progressVersion = '5',
                             query_file = 'query.txt',
                             csvTrigger=True,
                             exists=False, drop=True, backupS3=False):
    #Build stage on snowflake & query for all data
    #exists: if True, doesn't requery but uses existing stage
    #drop: if True, drops/cleans up stage on snowflake
    #csvTrigger=True, downloads as csv, otherwise parquet/snappy

    USER = getUsername()
    PASSWORD = getcode()
    ACCOUNT='[REDACTED]'
    DATABASE = '[REDACTED]'
    
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
        
        print('establishing warehouse and db...')
        cs.execute("USE warehouse COMPUTE_WH;")
        cs.execute("USE DATABASE [REDACTED];")
        
        print('loading ETL query from ' + query_file)        
        #snowflake query that works
        with open(query_file, 'r') as file:
            etl_query = file.read()


            
        print('fetch a small sample for the column labelling...')
        # Create a cursor object
        # Execute a SQL query
        cs.execute(etl_query+' LIMIT 1;')

        # Fetch column names from the cursor description
        column_names = [column[0] for column in cs.description]

        column_file='column_names.txt'
        #check and delete file if exists
        import os
        if os.path.exists(column_file):
            os.remove(column_file)

        # Write column names to a file
        with open(column_file, 'w') as f:
            f.write('\n'.join(column_names))
            
        print('mini query done. column names saved.')
        
        
        
        print('Creating stage: '+stage+' on Snowflake and filling it up with data')
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

        
        query = str("copy into @" + stage +
             "/result/data_ from ( " + 
             etl_query + 
                    " ) FILE_FORMAT = " +
                    ["(TYPE = CSV FIELD_OPTIONALLY_ENCLOSED_BY = '\"' FIELD_DELIMITER = ';' COMPRESSION = 'gzip')" if csvTrigger else "(TYPE = PARQUET COMPRESSION = 'SNAPPY')"][0] + ";")

        print('submitting query: '+ query)
        cs.execute(query)

        #datecutoff = '2019-08-09'
        #cs.execute("copy into @"+stage+"/result/data_ from (select answers.student_id, answers.session_id, answers.created_at, answers.math_question_id, answers.correctness, skill_list.skill_or_subskill_id as RL_TOP_LEVEL_SKILL_ID from [REDACTED].views.math_answers_fact as answers JOIN [REDACTED].prod.math_question_skills as skill_list ON answers.MATH_QUESTION_ID = skill_list.MATH_QUESTION_ID WHERE removed='FALSE' AND answers.created_at > "+ datecutoff +" SORT BY answers.created_at) file_format=(compression='gzip');")

        print('Completed query and stage copy: '+ str(time.time()-starttime) + ' seconds.')




    #pull down snowflake stage to local drive
    #make local dir
    import os
    local_dir = os.getcwd()
    
    if not os.path.exists(datastore):
        print('making new local dir')
        os.mkdir(datastore)
    else:
        #rename old folder using todays date as archive
        print('old directory found, renaming and making new local dir')
        import time
        suffix = str(int(time.time()))
        os.rename(datastore,datastore+suffix)
        #make new dir
        os.mkdir(datastore)  
        time.sleep(1)

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
    local_dir = os.getcwd()
    cs.execute("GET @"+stage +" file://"+local_dir+'//'+datastore+"//;")
    print('Completed transfer from '+stage + 'to ' +local_dir+'//'+datastore+'//'+ str(time.time()-starttime) + ' seconds.')



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

        for file in os.listdir('StudentProficiencyData'):
            if file[-6:]=='csv.gz':
                response=s3_client.upload_file('.//'+datastore+'//'+file,bucket,datastore+currentdate+'/'+file)
            else:
                print(file, ' skipped.')
        print('Transferred files from '+'.//'+datastore+'//'+file+' to '+bucket+datastore+currentdate+'/'+file+'.')

def run_snowflake_query(query):
    # Snowflake connection details
    USER = getUsername()
    PASSWORD = getcode()
    ACCOUNT = '[REDACTED]'
    DATABASE = '[REDACTED]'

    # Establish connection
    ctx = snowflake.connector.connect(
        user=USER,
        password=PASSWORD,
        account=ACCOUNT,
        database=DATABASE
    )

    # Create cursor
    cs = ctx.cursor()

    try:
        # Set warehouse and database
        cs.execute("USE warehouse COMPUTE_WH;")
        cs.execute("USE DATABASE [REDACTED];")

        # Execute the provided query
        cs.execute(query)

        # Fetch results
        results = cs.fetchall()

        # Get column names
        column_names = [col[0] for col in cs.description]

        return column_names, results

    finally:
        # Close cursor and connection
        cs.close()
        ctx.close()

def run_snowflake_query_async(query, check_interval=10):
    # Snowflake connection details
    USER = getUsername()
    PASSWORD = getcode()
    ACCOUNT = '[REDACTED]'
    DATABASE = '[REDACTED]'

    # Establish connection
    ctx = snowflake.connector.connect(
        user=USER,
        password=PASSWORD,
        account=ACCOUNT,
        database=DATABASE
    )

    # Create cursor
    cs = ctx.cursor()

    try:
        # Set warehouse and database
        cs.execute("USE warehouse COMPUTE_WH;")
        cs.execute("USE DATABASE [REDACTED];")

        # Execute the query asynchronously
        cs.execute_async(query)
        query_id = cs.sfqid  # Get the query ID

        # Check query status until it's done
        while True:
            try:
                cs.get_results_from_sfqid(query_id)
                print("Query status: SUCCESS")
                return "SUCCESS"
            except snowflake.connector.errors.ProgrammingError as e:
                if "not finished" in str(e):
                    print("Query status: RUNNING")
                else:
                    print(f"Query status: FAILED - {str(e)}")
                    return "FAILED"
            
            time.sleep(check_interval)

    finally:
        # Close cursor and connection
        cs.close()
        ctx.close()
        
        
def run_snowflake_queries_sequentially(queries):
    # Snowflake connection details
    USER = getUsername()
    PASSWORD = getcode()
    ACCOUNT = '[REDACTED]'
    DATABASE = '[REDACTED]'

    # Establish connection
    ctx = snowflake.connector.connect(
        user=USER,
        password=PASSWORD,
        account=ACCOUNT,
        database=DATABASE
    )

    # Create cursor
    cs = ctx.cursor()

    try:
        # Set warehouse and database
        cs.execute("USE warehouse COMPUTE_WH;")
        cs.execute("USE DATABASE [REDACTED];")

        for i, query in enumerate(queries, 1):
            print(f"Executing query {i}...")
            
            # Execute the query and wait for it to complete
            start_time = time.time()
            cs.execute(query)
            
            # Fetch all results to ensure query completion
            _ = cs.fetchall()
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"Query {i} completed in {duration:.2f} seconds.")

        print("All queries have been executed sequentially.")

    finally:
        # Close cursor and connection
        cs.close()
        ctx.close()
