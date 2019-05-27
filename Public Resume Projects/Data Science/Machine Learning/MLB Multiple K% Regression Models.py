from sklearn import linear_model, neighbors
from sklearn.ensemble import RandomForestRegressor
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
    gameDataFrame = pd.read_csv("Batter Stats with Previous Three Years Arranged No Minimum 2006-2018.csv")
    gameDataFrame = gameDataFrame.loc[gameDataFrame["YEAR"] >= 2009]
    gameDataFrame = gameDataFrame.loc[gameDataFrame["PA"] >= 75]
    gameDataFrame = gameDataFrame.dropna()
    trainBatterProjectedStrikeoutPercentKNN(gameDataFrame)

def trainBatterProjectedStrikeoutPercentLinearRegression(gameDataFrame):
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
    baseballModel = linear_model.LinearRegression()
    baseballModel = baseballModel.fit(trainDataInput, trainDataOutput)
    print("Mean Absolute Error of Test Data: " + str(np.mean(np.array(abs(np.array(baseballModel.predict(testDataInput)) - testDataOutput)))))

def trainBatterProjectedStrikeoutPercentKNN(gameDataFrame):
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
    baseballModel = neighbors.KNeighborsRegressor(n_neighbors=100)
    baseballModel = baseballModel.fit(trainDataInput, trainDataOutput)
    print("Mean Absolute Error of Test Data: " + str(np.mean(np.array(abs(np.array(baseballModel.predict(testDataInput)) - testDataOutput)))))

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