#the objective of this case is to prediction of the bike rental count on daily based on the environmental and seasonal settings

rm(list=ls())

#set working directory
setwd("C:/Users/Sunny/Desktop/edwisor/Bike rental")
getwd()



install.packages(c("ggplot2","corrgram","DMwR","caret","randomForest","C50","MASS","rpart","dplyr","plyr","reshape","ggplot2","data.table"))

library("ggplot2")
library("corrgram")
library("DMwR")
library("caret")
library("randomForest")
library("C50")
library("MASS")
library("rpart")
library("dplyr")
library("plyr")
library("reshape")
library("ggplot2")
library("data.table")


bike = read.csv("day.csv", header = TRUE, sep = ",")


head(bike)

dim(bike)

summary(bike)

str(bike)

#we can see variables like 'mnth',holiday','weekday','weathersit' are 
#catogical but taken as numeric values

#hence converting

bike$season=as.factor(bike$season)
bike$mnth=as.factor(bike$mnth)
bike$yr=as.factor(bike$yr)
bike$holiday=as.factor(bike$holiday)
bike$weekday=as.factor(bike$weekday)
bike$workingday=as.factor(bike$workingday)
bike$weathersit=as.factor(bike$weathersit)

str(bike)


#Nummeric  vaiables like 'temp','atem','hum','windspeed' are already Normalized 


missing_val = data.frame(apply(bike,2,function(x){sum(is.na(x))}))
missing_val

#no missing values found


#finding outliers

numeric_index = sapply(bike,is.numeric) #selecting only numeric

numeric_data = bike[,numeric_index]

cnames = colnames(numeric_data)
dim(numeric_data)

#loop to find outliers
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(bike))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "blue" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="cnt")+
           ggtitle(paste("Box plot of count for",cnames[i])))
}

gridExtra::grid.arrange(gn1,gn2,ncol=3)
gridExtra::grid.arrange(gn3,gn4,ncol=2)

#loop to remove all outliers from numeric data
numeric_data_no_outliers = numeric_data
  
for(i in cnames){
print(i)
val = numeric_data_no_outliers[,i][numeric_data_no_outliers[,i] %in% boxplot.stats(numeric_data_no_outliers[,i])$out]
print(length(val))
numeric_data_no_outliers = numeric_data_no_outliers[which(!numeric_data_no_outliers[,i] %in% val),]
} 

dim(numeric_data_no_outliers)

#merging and replacing data with no outlier numeric data

bike_ready = merge(x=bike, y=numeric_data_no_outliers, by="instant", all.y=TRUE)

dim(bike_ready)


#Feature Selection and dimension reduction
# Correlation Plot 
corrgram(bike[,numeric_index], order = F,upper.panel=panel.pie, text.panel=panel.txt, 
         main = "Correlation Plot")

#we see that temp and atemp are highly correlated and hence removing atemp column and 
#removing other redundent numeric coulumns after merging

bike = subset(bike_ready,select = -c(atemp.x,atemp.y,temp.x,hum.x,windspeed.x,casual.x,registered.x,cnt.x))
dim(bike)


#Modeling
#Linear Regression

#converting multilevel categorical variable into binary dummy variable
str(bike)
cnames= c("dteday","season","mnth","weekday","weathersit")
bike_lr=bike[,cnames]
dim(bike_lr)
target=data.frame(bike$cnt.y)
dim(target)
names(target)[1]="cnt"
bike_lr <- fastDummies::dummy_cols(bike_lr)
dim(bike_lr)
bike_lr= subset(bike_lr,select = -c(dteday,season,mnth,weekday,weathersit))
dim(bike_lr)
data = cbind(bike_lr,bike)
View(data)
data= subset(data,select = -c(dteday,season,mnth,weekday,weathersit))
dim(data)
bike_lr=cbind(data,target)
dim(bike_lr)

#dividind data into test and train
train_index = sample(1:nrow(bike_lr), 0.8 * nrow(bike_lr))
bike_train_lr = bike_lr[train_index,]
bike_test_lr = bike_lr[-train_index,]

#Linear regression model making
lm_model = lm(cnt ~., data = bike_train_lr)
summary(lm_model)

predictions_LR = predict(lm_model,bike_test_lr[,-768])

bike = subset(bike,select = -c(instant,dteday))
dim(bike)
#Decision tree regression  
train_index = sample(1:nrow(bike), 0.8 * nrow(bike))
train = bike[train_index,]
test = bike[-train_index,]


DT_model = rpart(cnt.y ~ ., data = train, method = "anova")
summary(DT_model)
predictions_DT = predict(DT_model, test[,-13])

#Random Forest Model
RF_model = randomForest(cnt.y ~ ., train, importance = TRUE, ntree = 200)
summary(RF_model)

predictions_RF = predict(RF_model, test[,-13])
plot(RF_model)


#evaluating MApe value and finding the best model

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))*100
}

MAPE(bike_test_lr[,768],  predictions_LR)
#108.3069

MAPE(test[,13], predictions_DT)
#11.48524

MAPE(test[,13], predictions_RF)
#6.028095

