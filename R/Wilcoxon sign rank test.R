Csv_file <- 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Wilcoxon signed rank test/BloodPressure.csv'
Bp <- read.csv(Csv_file)

help(wilcox.test)
#1
wilcox.test(Bp[,2], Bp[,3],
            mu = 0,
            alternative = 'less',
            paired = TRUE,
            conf.level = 0.95,
            exact = FALSE)


#2
Before <- c(135, 142, 137, 122, 147, 151, 131, 117, 154, 143, 133)
After <- c(127, 145, 131, 125, 132, 147, 119, 125, 132, 139, 122)

wilcox.test(Before, After,
            mu = 0,
            alternative = 'greater',
            paired = TRUE,
            conf.level = 0.95,
            exact = FALSE)
