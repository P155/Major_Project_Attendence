from database_conn import*
from LiveNFaceCNN import Start_camera
from Test_speech_rec_model import*
from playsound import playsound

start_rec()
Feature_extract()
label_voice= read_csv_load_model()
if label_voice =="Puja":
    print("voice Authorized")
    playsound('alert.mp3',True)
    mylist=Start_camera()
    playsound('thankyou.mp3',True)
    Post_to_database(mylist)
    print('connected to database')

    print('data Posted to database')
else:
    print("voice doesnot match")