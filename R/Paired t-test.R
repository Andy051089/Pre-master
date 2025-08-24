csv_file <- ('C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Paired t-test/BloodPressure.csv')
BP<- read.csv(csv_file)

dim(BP)
BP[1:3,]

?boxplot
boxplot(BP[,2:3])

# H0: The mean difference on BP is 0, HA: The mean difference on BP is < 0
?t.test
t.test(BP[,2], BP[,3], 
       mu = 0, 
       alternative = 'less', 
       paired = T, 
       conf.level = 0.95)
