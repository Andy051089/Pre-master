install.packages('dunn.test')
install.packages('FSA')

library(dunn.test)
library(FSA)

csv_file <- 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Kruskal Wallis/DietWeightLoss.csv'
data <- read.csv(csv_file)
attach(data)

kruskal.test(WeightLoss ~ Diet)

dunn.test(WeightLoss, Diet, method = 'Bonferroni', altp = T)