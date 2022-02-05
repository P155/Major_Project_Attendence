import psycopg2
from datetime import datetime,date
now = datetime.now()
today = date.today()
my_list={'Puja':'real','Ram':'fake'}
present = "P"
def Post_to_database(my_list):
	#establishing the connection
	con = psycopg2.connect(
	database="student_info", user='postgres', password='1234', host='localhost', port= '5432'
	)
	#Creating a cursor object using the cursor() method
	cursor = con.cursor()
	command=""" DROP TABLE IF EXISTS Student;"""
	command1 = """Create Table Student(
		name Char(50),
		liveness Char(50),
		attendence Char(2)
	);"""
	cursor.execute(command)
	cursor.execute(command1)
	for key,value in my_list.items():
		sql="""UPDATE classroom_studentattend SET attendence = 'P',
		date = '%s',
		datetime = '%s'
		WHERE first_name = '%s' ;"""%(now,today,key)
		sql1="""INSERT INTO Student (name,liveness,attendence)             
              VALUES(%s,%s,%s)"""
		data_to_insert=(key,value,present)
		cursor.execute(sql)
		cursor.execute(sql1,data_to_insert)
	con.commit()
	con.close()

Post_to_database(my_list)
