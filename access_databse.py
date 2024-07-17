import sqlite3

class AccessDatabase:
    def __init__(self):
        self.con = sqlite3.connect('StoredFaces.db') #create a database, establish connection to it
        self.cur = self.con.cursor() #cursor allws us to access and edit the database
    
    def delete_person(self, person):
        self.cur.execute('''SELECT * FROM embeddings''')
        rows = self.cur.fetchall() #list of the all the rows in the database, info on every person
        for row in rows:
            print(str(row[0]), end=" ")
        print('')
            
        self.cur.execute(
            "DELETE FROM embeddings WHERE person=?", (person, )
        )
        self.con.commit()
        
        self.cur.execute('''SELECT * FROM embeddings''')
        rows = self.cur.fetchall() #list of the all the rows in the database, info on every person
        for row in rows:
            print(str(row[0]), end=" ")

database = AccessDatabase()
database.delete_person('Charlie')