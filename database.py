import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Emotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    date = db.Column(db.Date, default=datetime.datetime.today(), nullable=True)
    image = db.Column(db.LargeBinary)
    emotion = db.Column(db.Enum('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', ))
    rate = db.Column(db.Float, default=0.0)  #
    # emotion_id = db.Column(db.Integer, db.ForeignKey('emotion_type.id'), ondelete='PROTECT')


def get_emotions():
    emotions = db.session.query(Emotion.date, Emotion.emotion, Emotion.rate,
            db.func.count(Emotion.emotion).label('count'), ).group_by(Emotion.date, Emotion.emotion
            ).all()
    # for result in emotions:
    #     print(result.date, result.emotion, result.rate, result.count)
    return emotions

def create_data():
    # today = datetime.date.today()
    # first_day_of_year = datetime.date(today.year - 1, 1, 1)
    # i = 0
    # for date in range((today - first_day_of_year).days + 1):
    #     # print("(dates):", date)
    #     print("(dates):", first_day_of_year + datetime.timedelta(days=date))
    #     for item in ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', ]:
    #         emotion = Emotion(date=first_day_of_year + datetime.timedelta(days=date),
    #                 emotion=random.choice(['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', ]),
    #                 rate=random.random()
    #                 )
    #         emotion2 = Emotion(date=first_day_of_year + datetime.timedelta(days=date), emotion=item,
    #                 rate=random.random()
    #                 )
    #         emotion3 = Emotion(date=first_day_of_year + datetime.timedelta(days=date),
    #                 emotion=random.choice(['Angry', 'Happy', 'Neutral', 'Sad', ]), rate=random.random()
    #                 )
    #         emotion4 = Emotion(date=first_day_of_year + datetime.timedelta(days=date),
    #                 emotion=random.choice(['Happy', 'Neutral', ]), rate=random.random()
    #                 )
    #
    #     db.session.add(emotion)
    #     db.session.add(emotion2)
    #     db.session.add(emotion3)
    #     db.session.add(emotion4)
    #     i += 1
    #     if i % 100 == 0:
    #         db.session.commit()
    # db.session.commit()
    
    # print("data", data)
    # emotion_types = [emotion.to_dict() for emotion in data]
    # data = Emotion.query.all()
    # for emo in data:
    #     print("emo", emo.emotion)
    results = db.session.query(Emotion.date, Emotion.emotion, Emotion.rate,
            db.func.count(Emotion.emotion).label('count'), ).group_by(Emotion.date, Emotion.emotion
            ).all()
    for result in results:
        print(result.date, result.emotion, result.count)
