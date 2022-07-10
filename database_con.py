import imp
import pymysql
from datetime import date,datetime
#database connection
conn= pymysql.connect(host="localhost",user="root",passwd="",database="npd" )
cursor = conn.cursor()
# some other statements  with the help of cursor
def dbEntry(num_plate):
    sql = ("INSERT INTO NUMBERS(Number,date,time) VALUES (%s,%s,%s)")
    conn.ping() 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    cursor.execute(sql, (num_plate,date.today(),current_time))
    conn.commit()
    conn.close()  
