csv_file <- 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/One Way ANOVA/DietWeightLoss.csv'
data <- read.csv(csv_file)

head(data)
attach(data)
boxplot(WeightLoss~ Diet)

ans <- aov(WeightLoss~ Diet)
summary(ans)

TukeyHSD(ans)

plot(TukeyHSD(ans))

P<0.05跟別人不一樣