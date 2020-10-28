

library(alr4)

head(Heights)

x <- Heights$mheight
y <- Heights$dheight

x0 <- (x - mean(x)) / sqrt(var(x))

x1 <- (y - mean(x)) / sqrt(var(x))


qqnorm(x0)
qqline(x0, col='blue', lwd=2)


df <- data.frame(x, y)

write.csv(df, 'Desktop/Polygenic/github-repo/data/person-lee.csv', row.names=FALSE)
