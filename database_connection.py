import numpy as np 
import psycopg2 as psy 
import pickle

class DatabaseConnection:
    def __init__(self,  db_name: str, table_name: str):
        self.db_name = db_name
        self.table_name = table_name

    def connect(self):
        db_connect_kwargs = {
            'dbname': self.db_name,
            'user': 'fifi',
            'password': 'anx00di1',
            'host': 'localhost',
            'port': '5432'
        }

        self.connection = psy.connect(**db_connect_kwargs)
        self.cursor = self.connection.cursor()

        return self

    def execute_query(self, video_id: str, markers: np.ndarray, video: np.ndarray, audio: np.ndarray):
        self.cursor.execute(
            f"""
            INSERT INTO {self.table_name} (video_id, markers, video, audio)
            VALUES (%s, %s, %s, %s)
            """,
            (video_id, 
             pickle.dumps(markers, protocol=pickle.HIGHEST_PROTOCOL), 
             pickle.dumps(video, protocol=pickle.HIGHEST_PROTOCOL), 
             pickle.dumps(audio, protocol=pickle.HIGHEST_PROTOCOL))
        )
        self.connection.commit()

    def fetch_data(self, video_id: str):
        self.cursor.execute(
            f"""
            SELECT markers, video, audio FROM {self.table_name}
            WHERE video_id=%s
            """,
            (video_id,)
        )
        fetch = self.cursor.fetchone()
        return pickle.loads(fetch[0]), pickle.loads(fetch[1]), pickle.loads(fetch[2])
    
    def ifExists(self, video_id: str):
        self.cursor.execute(
            f"""
            SELECT COUNT(*) FROM {self.table_name}
            WHERE video_id=%s
            """,
            (video_id,)
        )
        return self.cursor.fetchone()[0]