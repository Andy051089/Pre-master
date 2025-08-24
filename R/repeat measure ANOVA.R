install.packages("ez")
library(ez)

data <- data.frame(
  subject = factor(rep(1:10, each=3)),
  time = factor(rep(c("T1", "T2", "T3"), times=10)),
  score = c(78, 84, 90, 79, 83, 85, 82, 89, 91, 80, 84, 86, 81, 88, 89, 83, 85, 87, 84, 89, 90, 85, 87, 88, 86, 90, 91, 87, 89, 90)
)
attach(data)

ans <- ezANOVA(
  data = data,
  dv = score,
  wid = subject,
  within = time,
)

result <- pairwise.t.test(x = score, g = time, paired = T, 
                p.adjust.method = 'bonf')

print(result)