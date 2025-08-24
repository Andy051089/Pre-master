csv_file <- 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Chi square test/LungCapData.csv'
data <- read.csv(csv_file)

head(data)
attach(data)

table <- table(Gender, Smoke)
barplot(table, beside = T, legend.text = T)
chisq.test(table)

