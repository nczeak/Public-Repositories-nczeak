from sklearn import linear_model, neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, BatchNormalization, Activation, LeakyReLU
import matplotlib.pyplot as plotFunctions
import seaborn as sns
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def mainFunction():
    gameDataFrame = pd.read_csv("Cleaned NHL play-by-play 2013-2018.csv")
    trainExpectedGoalsKNN(gameDataFrame)

def cleanPlayDataFrame():
    gameDataFrame = pd.read_csv("NHL play-by-play 2013-2014.csv")
    for thisYear in range(2014, 2018):
        print("\t" + "Merging " + str(thisYear) + "-" + str(thisYear + 1) + " season.")
        gameDataFrame = gameDataFrame.merge(pd.read_csv("NHL play-by-play " + str(thisYear) + "-" + str(thisYear + 1) + ".csv"), on=list(gameDataFrame), how="outer")

    gameDataFrameColumns = list(gameDataFrame)
    unnamedColumnNumber = 0
    while True:
        try:
            gameDataFrameColumns.remove("Unnamed: " + str(unnamedColumnNumber))
            unnamedColumnNumber += 1
        except ValueError:
            break
    gameDataFrameColumns.remove("Game_Id")
    gameDataFrameColumns.remove("Date")
    gameDataFrameColumns.remove("p1_name")
    gameDataFrameColumns.remove("p1_ID")
    gameDataFrameColumns.remove("p2_name")
    gameDataFrameColumns.remove("p2_ID")
    gameDataFrameColumns.remove("p3_name")
    gameDataFrameColumns.remove("p3_ID")
    gameDataFrameColumns.remove("Away_Players")
    gameDataFrameColumns.remove("Home_Players")
    gameDataFrameColumns.remove("Away_Goalie")
    gameDataFrameColumns.remove("Away_Goalie_Id")
    gameDataFrameColumns.remove("Home_Goalie")
    gameDataFrameColumns.remove("Home_Goalie_Id")
    gameDataFrameColumns.remove("Away_Coach")
    gameDataFrameColumns.remove("Home_Coach")
    for thisTeam in ["away", "home"]:
        for thisPlayer in range(1, 7):
            gameDataFrameColumns.remove(thisTeam + "Player" + str(thisPlayer))
            gameDataFrameColumns.remove(thisTeam + "Player" + str(thisPlayer) + "_id")
    gameDataFrame = gameDataFrame[gameDataFrameColumns]

    desiredEvents = ["GOAL", "SHOT", "MISS"]
    undesiredEvents = list(set(gameDataFrame["Event"]) - set(desiredEvents))
    for thisEvent in undesiredEvents:
        gameDataFrame = gameDataFrame.loc[gameDataFrame["Event"] != thisEvent]

    gameDataFrame["Shot Distance"] = gameDataFrame.apply(getShotDistance, axis = 1)
    print(list(set(list(gameDataFrame["Type"]))))
    print(list(sorted(set(list(gameDataFrame["Strength"])))))
    undesiredStrengths = ["0x0", "1x0", "0x1", "0x5", "5x0", "1x5", "5x1", "3x2", "2x3", "5x10", "10x5", "7x5", "5x7", "8x5", "5x8", "4x1", "1x4", "5x2", "2x5"]
    for thisStrength in undesiredStrengths:
        gameDataFrame = gameDataFrame.loc[gameDataFrame["Event"] != thisStrength]

    gameDataFrame = gameDataFrame.dropna()
    gameDataFrame["Tip-In"] = gameDataFrame["Type"] == "TIP-IN"
    gameDataFrame["Slap Shot"] = gameDataFrame["Type"] == "SLAP SHOT"
    gameDataFrame["Wrist Shot"] = gameDataFrame["Type"] == "WRIST SHOT"
    gameDataFrame["Backhand"] = gameDataFrame["Type"] == "BACKHAND"
    gameDataFrame["Wrap-Around"] = gameDataFrame["Type"] == "WRAP-AROUND"
    gameDataFrame["Snap Shot"] = gameDataFrame["Type"] == "SNAP SHOT"

    gameDataFrame["5v5"] = gameDataFrame["Strength"] == "5x5"
    gameDataFrame["4v4"] = gameDataFrame["Strength"] == "4x4"
    gameDataFrame["3v3"] = gameDataFrame["Strength"] == "3x3"
    gameDataFrame["Home 6v5"] = gameDataFrame["Strength"] == "6x5"
    gameDataFrame["Home 6v4"] = gameDataFrame["Strength"] == "6x4"
    gameDataFrame["Home 6v3"] = gameDataFrame["Strength"] == "6x3"
    gameDataFrame["Home 5v4"] = gameDataFrame["Strength"] == "5x4"
    gameDataFrame["Home 5v3"] = gameDataFrame["Strength"] == "5x3"
    gameDataFrame["Away 6v5"] = gameDataFrame["Strength"] == "5x6"
    gameDataFrame["Away 6v4"] = gameDataFrame["Strength"] == "4x6"
    gameDataFrame["Away 6v3"] = gameDataFrame["Strength"] == "3x6"
    gameDataFrame["Away 5v4"] = gameDataFrame["Strength"] == "4x5"
    gameDataFrame["Away 5v3"] = gameDataFrame["Strength"] == "3x5"

    gameDataFrame["Home Team Shot"] = gameDataFrame["Ev_Team"] == gameDataFrame["Home_Team"]
    gameDataFrame["Goal"] = gameDataFrame["Event"] == "GOAL"
    gameDataFrame.to_csv("Cleaned NHL play-by-play 2013-2018.csv")

def getShotDistance(thisRow):
    try:
        return int(str(thisRow["Description"]).split(" ft.")[0].split(" ")[-1])
    except ValueError:
        return None

def trainExpectedGoalsLogisticRegression(gameDataFrame):
    gameDataFrame = gameDataFrame[["Shot Distance",
                                   "Home Team Shot",
                                   "Tip-In",
                                   "Slap Shot",
                                   "Wrist Shot",
                                   "Backhand",
                                   "Wrap-Around",
                                   "Snap Shot",
                                   "4v4",
                                   "3v3",
                                   "Away 6v5",
                                   "Away 6v4",
                                   "Away 6v3",
                                   "Away 5v4",
                                   "Away 5v3",
                                   "Home 6v5",
                                   "Home 6v4",
                                   "Home 6v3",
                                   "Home 5v4",
                                   "Home 5v3",
                                   "Goal"]]
    trainData = gameDataFrame.sample(frac=0.75)
    testData = gameDataFrame.loc[~ gameDataFrame.index.isin(trainData.index)]
    trainDataInput = trainData[["Shot Distance",
                                   "Home Team Shot",
                                   "Tip-In",
                                   "Slap Shot",
                                   "Wrist Shot",
                                   "Backhand",
                                   "Wrap-Around",
                                   "Snap Shot",
                                   "4v4",
                                   "3v3",
                                   "Away 6v5",
                                   "Away 6v4",
                                   "Away 6v3",
                                   "Away 5v4",
                                   "Away 5v3",
                                   "Home 6v5",
                                   "Home 6v4",
                                   "Home 6v3",
                                   "Home 5v4",
                                   "Home 5v3"]]
    testDataInput = testData[["Shot Distance",
                                   "Home Team Shot",
                                   "Tip-In",
                                   "Slap Shot",
                                   "Wrist Shot",
                                   "Backhand",
                                   "Wrap-Around",
                                   "Snap Shot",
                                   "4v4",
                                   "3v3",
                                   "Away 6v5",
                                   "Away 6v4",
                                   "Away 6v3",
                                   "Away 5v4",
                                   "Away 5v3",
                                   "Home 6v5",
                                   "Home 6v4",
                                   "Home 6v3",
                                   "Home 5v4",
                                   "Home 5v3"]]
    trainDataOutput = trainData[["Goal"]]
    testDataOutput = testData[["Goal"]]
    trainDataInput = np.array(trainDataInput).reshape(len(trainDataInput), 20)
    testDataInput = np.array(testDataInput).reshape(len(testDataInput), 20)
    trainDataOutput = np.array(trainData[["Goal"]]).reshape(len(trainData))
    testDataOutput = np.array(testData[["Goal"]]).reshape(len(testData))
    baseballModel = linear_model.LogisticRegression()
    baseballModel = baseballModel.fit(trainDataInput, trainDataOutput)

    print("Log Loss of Test Data: " + str(log_loss(trainDataOutput, np.array(baseballModel.predict_proba(trainDataInput)))))

def trainExpectedGoalsKNN(gameDataFrame):
    gameDataFrame = gameDataFrame[["Shot Distance",
                                   "Home Team Shot",
                                   "Tip-In",
                                   "Slap Shot",
                                   "Wrist Shot",
                                   "Backhand",
                                   "Wrap-Around",
                                   "Snap Shot",
                                   "4v4",
                                   "3v3",
                                   "Away 6v5",
                                   "Away 6v4",
                                   "Away 6v3",
                                   "Away 5v4",
                                   "Away 5v3",
                                   "Home 6v5",
                                   "Home 6v4",
                                   "Home 6v3",
                                   "Home 5v4",
                                   "Home 5v3",
                                   "Goal"]]
    trainData = gameDataFrame.sample(frac=0.75)
    testData = gameDataFrame.loc[~ gameDataFrame.index.isin(trainData.index)]
    trainDataInput = trainData[["Shot Distance",
                                "Home Team Shot",
                                "Tip-In",
                                "Slap Shot",
                                "Wrist Shot",
                                "Backhand",
                                "Wrap-Around",
                                "Snap Shot",
                                "4v4",
                                "3v3",
                                "Away 6v5",
                                "Away 6v4",
                                "Away 6v3",
                                "Away 5v4",
                                "Away 5v3",
                                "Home 6v5",
                                "Home 6v4",
                                "Home 6v3",
                                "Home 5v4",
                                "Home 5v3"]]
    testDataInput = testData[["Shot Distance",
                              "Home Team Shot",
                              "Tip-In",
                              "Slap Shot",
                              "Wrist Shot",
                              "Backhand",
                              "Wrap-Around",
                              "Snap Shot",
                              "4v4",
                              "3v3",
                              "Away 6v5",
                              "Away 6v4",
                              "Away 6v3",
                              "Away 5v4",
                              "Away 5v3",
                              "Home 6v5",
                              "Home 6v4",
                              "Home 6v3",
                              "Home 5v4",
                              "Home 5v3"]]
    trainDataOutput = trainData[["Goal"]]
    testDataOutput = testData[["Goal"]]
    trainDataInput = np.array(trainDataInput).reshape(len(trainDataInput), 20)
    testDataInput = np.array(testDataInput).reshape(len(testDataInput), 20)
    trainDataOutput = np.array(trainData[["Goal"]]).reshape(len(trainData))
    testDataOutput = np.array(testData[["Goal"]]).reshape(len(testData))
    baseballModel = neighbors.KNeighborsClassifier(n_neighbors=100)
    baseballModel = baseballModel.fit(trainDataInput, trainDataOutput)
    print("Log Loss of Test Data: " + str(log_loss(trainDataOutput, np.array(baseballModel.predict_proba(trainDataInput)))))

def trainBatterProjectedStrikeoutPercentRandomForest(gameDataFrame):
    gameDataFrame = gameDataFrame[["AGE",
                                   "First Year MLB Played",
                                   "First Year AAA Played",
                                   "First Year AA Played",
                                   "First Year A+ Played",
                                   "First Year A Played",
                                   "First Year A- Played",
                                   "First Year AGE",
                                   "First Year K%",
                                   "Second Year MLB Played",
                                   "Second Year AAA Played",
                                   "Second Year AA Played",
                                   "Second Year A+ Played",
                                   "Second Year A Played",
                                   "Second Year A- Played",
                                   "Second Year AGE",
                                   "Second Year K%",
                                   "Third Year MLB Played",
                                   "Third Year AAA Played",
                                   "Third Year AA Played",
                                   "Third Year A+ Played",
                                   "Third Year A Played",
                                   "Third Year A- Played",
                                   "Third Year AGE",
                                   "Third Year K%",
                                   "K%"]]
    trainData = gameDataFrame.sample(frac=0.75)
    testData = gameDataFrame.loc[~ gameDataFrame.index.isin(trainData.index)]
    trainDataInput = trainData[["AGE",
                                "First Year MLB Played",
                                "First Year AAA Played",
                                "First Year AA Played",
                                "First Year A+ Played",
                                "First Year A Played",
                                "First Year A- Played",
                                "First Year AGE",
                                "First Year K%",
                                "Second Year MLB Played",
                                "Second Year AAA Played",
                                "Second Year AA Played",
                                "Second Year A+ Played",
                                "Second Year A Played",
                                "Second Year A- Played",
                                "Second Year AGE",
                                "Second Year K%",
                                "Third Year MLB Played",
                                "Third Year AAA Played",
                                "Third Year AA Played",
                                "Third Year A+ Played",
                                "Third Year A Played",
                                "Third Year A- Played",
                                "Third Year AGE",
                                "Third Year K%"]]
    testDataInput = testData[["AGE",
                              "First Year MLB Played",
                              "First Year AAA Played",
                              "First Year AA Played",
                              "First Year A+ Played",
                              "First Year A Played",
                              "First Year A- Played",
                              "First Year AGE",
                              "First Year K%",
                              "Second Year MLB Played",
                              "Second Year AAA Played",
                              "Second Year AA Played",
                              "Second Year A+ Played",
                              "Second Year A Played",
                              "Second Year A- Played",
                              "Second Year AGE",
                              "Second Year K%",
                              "Third Year MLB Played",
                              "Third Year AAA Played",
                              "Third Year AA Played",
                              "Third Year A+ Played",
                              "Third Year A Played",
                              "Third Year A- Played",
                              "Third Year AGE",
                              "Third Year K%"]]
    trainDataOutput = trainData[["K%"]]
    testDataOutput = testData[["K%"]]
    trainDataInput = np.array(trainDataInput).reshape(len(trainDataInput), 25)
    testDataInput = np.array(testDataInput).reshape(len(testDataInput), 25)
    trainDataOutput = np.array(trainData[["K%"]]).reshape(len(trainData))
    testDataOutput = np.array(testData[["K%"]]).reshape(len(testData))
    baseballModel = RandomForestRegressor(n_estimators = 100)
    baseballModel = baseballModel.fit(trainDataInput, trainDataOutput)
    print("Mean Absolute Error of Test Data: " + str(np.mean(np.array(abs(np.array(baseballModel.predict(testDataInput)) - testDataOutput)))))

def trainBatterProjectedStrikeoutPercentKeras(gameDataFrame):
    gameDataFrame = gameDataFrame[["AGE",
                                   "First Year MLB Played",
                                   "First Year AAA Played",
                                   "First Year AA Played",
                                   "First Year A+ Played",
                                   "First Year A Played",
                                   "First Year A- Played",
                                   "First Year AGE",
                                   "First Year K%",
                                   "Second Year MLB Played",
                                   "Second Year AAA Played",
                                   "Second Year AA Played",
                                   "Second Year A+ Played",
                                   "Second Year A Played",
                                   "Second Year A- Played",
                                   "Second Year AGE",
                                   "Second Year K%",
                                   "Third Year MLB Played",
                                   "Third Year AAA Played",
                                   "Third Year AA Played",
                                   "Third Year A+ Played",
                                   "Third Year A Played",
                                   "Third Year A- Played",
                                   "Third Year AGE",
                                   "Third Year K%",
                                   "K%"]]
    while True:
        trainData = gameDataFrame.sample(frac=0.75)
        testData = gameDataFrame.loc[~ gameDataFrame.index.isin(trainData.index)]
        trainDataInput = trainData[["AGE",
                                    "First Year MLB Played",
                                    "First Year AAA Played",
                                    "First Year AA Played",
                                    "First Year A+ Played",
                                    "First Year A Played",
                                    "First Year A- Played",
                                    "First Year AGE",
                                    "First Year K%",
                                    "Second Year MLB Played",
                                    "Second Year AAA Played",
                                    "Second Year AA Played",
                                    "Second Year A+ Played",
                                    "Second Year A Played",
                                    "Second Year A- Played",
                                    "Second Year AGE",
                                    "Second Year K%",
                                    "Third Year MLB Played",
                                    "Third Year AAA Played",
                                    "Third Year AA Played",
                                    "Third Year A+ Played",
                                    "Third Year A Played",
                                    "Third Year A- Played",
                                    "Third Year AGE",
                                    "Third Year K%"]]
        testDataInput = testData[["AGE",
                                  "First Year MLB Played",
                                  "First Year AAA Played",
                                  "First Year AA Played",
                                  "First Year A+ Played",
                                  "First Year A Played",
                                  "First Year A- Played",
                                  "First Year AGE",
                                  "First Year K%",
                                  "Second Year MLB Played",
                                  "Second Year AAA Played",
                                  "Second Year AA Played",
                                  "Second Year A+ Played",
                                  "Second Year A Played",
                                  "Second Year A- Played",
                                  "Second Year AGE",
                                  "Second Year K%",
                                  "Third Year MLB Played",
                                  "Third Year AAA Played",
                                  "Third Year AA Played",
                                  "Third Year A+ Played",
                                  "Third Year A Played",
                                  "Third Year A- Played",
                                  "Third Year AGE",
                                  "Third Year K%"]]
        trainDataOutput = trainData[["K%"]]
        testDataOutput = testData[["K%"]]
        trainDataInput = np.array(trainDataInput).reshape(len(trainDataInput), 25, 1)
        testDataInput = np.array(testDataInput).reshape(len(testDataInput), 25, 1)
        trainDataOutput = np.array(trainData[["K%"]]).reshape(len(trainData), 1)
        testDataOutput = np.array(testData[["K%"]]).reshape(len(testData), 1)

        dropoutAmount = 0.3
        baseballModel = Sequential()
        baseballModel.add(Flatten())
        baseballModel.add(Dense(8))
        baseballModel.add(LeakyReLU(alpha=0.3))
        baseballModel.add(Dropout(dropoutAmount))
        baseballModel.add(Dense(8))
        baseballModel.add(LeakyReLU(alpha=0.3))
        baseballModel.add(Dropout(dropoutAmount))
        baseballModel.add(Dense(1))

        baseballModel.compile(loss=keras.losses.mean_absolute_error,
                              optimizer=keras.optimizers.Adadelta(),
                              metrics=['mae'])

        print("Training Batter K% Projection Model on " + str(len(trainDataInput)) + " samples:")
        baseballModel.fit(trainDataInput,
                          trainDataOutput,
                          batch_size=128,
                          epochs=1000,
                          verbose=0,
                          validation_data=(testDataInput, testDataOutput))
        modelScoreTrain = baseballModel.evaluate(trainDataInput, trainDataOutput, verbose=0)
        modelScoreTest = baseballModel.evaluate(testDataInput, testDataOutput, verbose=0)
        modelStandardDeviation = np.std(np.array(baseballModel.predict(np.array(gameDataFrame[["AGE",
                                                                                               "First Year MLB Played",
                                                                                               "First Year AAA Played",
                                                                                               "First Year AA Played",
                                                                                               "First Year A+ Played",
                                                                                               "First Year A Played",
                                                                                               "First Year A- Played",
                                                                                               "First Year AGE",
                                                                                               "First Year K%",
                                                                                               "Second Year MLB Played",
                                                                                               "Second Year AAA Played",
                                                                                               "Second Year AA Played",
                                                                                               "Second Year A+ Played",
                                                                                               "Second Year A Played",
                                                                                               "Second Year A- Played",
                                                                                               "Second Year AGE",
                                                                                               "Second Year K%",
                                                                                               "Third Year MLB Played",
                                                                                               "Third Year AAA Played",
                                                                                               "Third Year AA Played",
                                                                                               "Third Year A+ Played",
                                                                                               "Third Year A Played",
                                                                                               "Third Year A- Played",
                                                                                               "Third Year AGE",
                                                                                               "Third Year K%"]]).reshape(
            len(gameDataFrame), 25, 1))))
        print("Standard Deviation: " + str(modelStandardDeviation))
        print("Standard Deviation of Actual K%: " + str(np.std(np.array(gameDataFrame["K%"]))))
        print("Train Loss: " + str(modelScoreTrain[0]))
        print("Test Loss: " + str(modelScoreTest[0]))
        if ((modelScoreTrain[0] <= 3) & (modelScoreTest[0] <= 3)):
            print("Desired Loss reached.")
        if ((modelScoreTrain[0] <= 3) & (modelScoreTest[0] <= 3) & (modelStandardDeviation >= 4.5)):
            break
        print("Model re-training.\n")
    print("Standard Deviation of Actual K%: " + str(np.std(np.array(gameDataFrame["K%"]))))
    print("Standard Deviation of Projected K%: " + str(np.std(np.array(baseballModel.predict(np.array(gameDataFrame[["AGE",
                                                                                                                     "First Year MLB Played",
                                                                                                                     "First Year AAA Played",
                                                                                                                     "First Year AA Played",
                                                                                                                     "First Year A+ Played",
                                                                                                                     "First Year A Played",
                                                                                                                     "First Year A- Played",
                                                                                                                     "First Year AGE",
                                                                                                                     "First Year K%",
                                                                                                                     "Second Year MLB Played",
                                                                                                                     "Second Year AAA Played",
                                                                                                                     "Second Year AA Played",
                                                                                                                     "Second Year A+ Played",
                                                                                                                     "Second Year A Played",
                                                                                                                     "Second Year A- Played",
                                                                                                                     "Second Year AGE",
                                                                                                                     "Second Year K%",
                                                                                                                     "Third Year MLB Played",
                                                                                                                     "Third Year AAA Played",
                                                                                                                     "Third Year AA Played",
                                                                                                                     "Third Year A+ Played",
                                                                                                                     "Third Year A Played",
                                                                                                                     "Third Year A- Played",
                                                                                                                     "Third Year AGE",
                                                                                                                     "Third Year K%"]]).reshape(
        len(gameDataFrame), 25, 1))))))
    sns.distplot(np.array(baseballModel.predict(np.array(gameDataFrame[["AGE",
                                                                        "First Year MLB Played",
                                                                        "First Year AAA Played",
                                                                        "First Year AA Played",
                                                                        "First Year A+ Played",
                                                                        "First Year A Played",
                                                                        "First Year A- Played",
                                                                        "First Year AGE",
                                                                        "First Year K%",
                                                                        "Second Year MLB Played",
                                                                        "Second Year AAA Played",
                                                                        "Second Year AA Played",
                                                                        "Second Year A+ Played",
                                                                        "Second Year A Played",
                                                                        "Second Year A- Played",
                                                                        "Second Year AGE",
                                                                        "Second Year K%",
                                                                        "Third Year MLB Played",
                                                                        "Third Year AAA Played",
                                                                        "Third Year AA Played",
                                                                        "Third Year A+ Played",
                                                                        "Third Year A Played",
                                                                        "Third Year A- Played",
                                                                        "Third Year AGE",
                                                                        "Third Year K%"]]).reshape(len(gameDataFrame), 25, 1))), kde=False)
    plotFunctions.show()
    sns.distplot(np.array(gameDataFrame["K%"]), kde=False)
    plotFunctions.show()
    baseballModel.save("Batter K% Projection Model.h5")

def trainBatterProjectedStrikeoutPercentGBM(gameDataFrame):
    gameDataFrame = gameDataFrame[["AGE",
                                   "First Year MLB Played",
                                   "First Year AAA Played",
                                   "First Year AA Played",
                                   "First Year A+ Played",
                                   "First Year A Played",
                                   "First Year A- Played",
                                   "First Year AGE",
                                   "First Year K%",
                                   "Second Year MLB Played",
                                   "Second Year AAA Played",
                                   "Second Year AA Played",
                                   "Second Year A+ Played",
                                   "Second Year A Played",
                                   "Second Year A- Played",
                                   "Second Year AGE",
                                   "Second Year K%",
                                   "Third Year MLB Played",
                                   "Third Year AAA Played",
                                   "Third Year AA Played",
                                   "Third Year A+ Played",
                                   "Third Year A Played",
                                   "Third Year A- Played",
                                   "Third Year AGE",
                                   "Third Year K%",
                                   "K%"]]

def modelFinishedEmail():
    emailDetails = {}
    emailDetails["Sender"] = "zeakbettingbot@gmail.com"
    emailDetails["Recipients"] = ["nczeak@gmail.com"]
    emailDetails["Subject"] = "Model Finished Training"
    emailDetails["Sender Password"] = "theswamp1853"
    emailDetails["Body"] = "Your recent model has finished training. Please check your computer."
    emailDetails["Attachments"] = []
    sendEmail(emailDetails)

def sendEmail(emailDetails):
    for thisRecipient in emailDetails["Recipients"]:
        thisEmail = MIMEMultipart()
        thisEmail["From"] = "zeakbettingbot@gmail.com"
        thisEmail["To"] = thisRecipient
        thisEmail["Subject"] = emailDetails["Subject"]
        thisEmail.attach(MIMEText(emailDetails["Body"], "plain"))
        for thisAttachment in emailDetails["Attachments"]:
            if thisAttachment["Type"] == "Image":
                thisEmail.attach(MIMEImage(thisAttachment["File"].read(), name=thisAttachment["Name"]))

        thisSession = smtplib.SMTP("smtp.gmail.com", 587)
        thisSession.starttls()
        thisSession.login("zeakbettingbot@gmail.com", "theswamp1853")
        thisSession.sendmail("zeakbettingbot@gmail.com", thisRecipient, thisEmail.as_string())
        thisSession.quit()

if __name__ == "__main__":
    mainFunction()