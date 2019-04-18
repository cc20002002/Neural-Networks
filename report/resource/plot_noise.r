require('ggplot2')
library(ggthemes)
theme_set(theme_default())  # from ggthemes
require(data.table)
x=t(t(c(-20:20)))*5
y0=t(t(dnorm(x,mean=0,sd=80)))
y1=t(t(dnorm(x,mean=0,sd=sqrt(40))))
y2=t(t(dpois(x+40,lambda = 40)))
df=cbind(rbind(x,x,x),rbind(y0,y1,y2))
df=data.table(df)
df=cbind(df,'Gaussian')
df[1:41,3]='Gaussian noise N(0,6400)'
df[42:82,3]='Gaussian noise N(0,40) '
df[83:123,3]='Poisson noise'
names(df)=c('Noise','Density','Distributions')
ggplot(df, aes(x=Noise, y=Density,color=Distributions)) + 
  geom_line(alpha=1) + scale_color_manual(values=c("red", "blue",'green'))

ggsave('noise.pdf',width = 7,height = 4, units = "in")
