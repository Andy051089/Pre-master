install.packages('car')
library(car)

csv_file <- ('C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Two sample t-test/LungCapData.csv')
Data <- read.csv(csv_file)

head(Data)
?t.test

Data$LungCap[Data$Smoke == 'yes']

boxplot(Data$LungCap ~ Data$Smoke)

t.test(Data$LungCap ~ Data$Smoke,
       alternative = 'two.sided',
       conf.level = 0.95,
       paired = FALSE,
       var.equal = T)

t.test(Data$LungCap[Data$Smoke == 'no'], Data$LungCap[Data$Smoke == 'yes'],
       alternative = 'two.sided',
       conf.level = 0.95,
       paired = FALSE,
       var.equal = F)

var(Data$LungCap[Data$Smoke == 'no'])
var(Data$LungCap[Data$Smoke == 'yes'])

#Levene's test
#H0:變異數相同  HA:變異數不同

leveneTest(Data$LungCap ~ Data$Smoke)
