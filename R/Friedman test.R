install.packages('PMCMRplus')

library(PMCMRplus)

  csv_file <- 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Friedman/Friedman.csv'
data <- read.csv(csv_file)

View(data)

friedman.test(data$Rating, data$Tea, data$ID)

PMCMRplus::frdAllPairsNemenyiTest(data$Rating, 
                                  data$Tea, 
                                  data$ID, 
                                  p.adjust.methods = 'bonferroni')