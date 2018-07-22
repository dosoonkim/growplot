library('ggplot2')
library('ggthemes')
library('RColorBrewer')

data <- read.csv('growth_kinetic.csv', header=TRUE, sep=",")

plot <- ggplot(data, aes(x=time, y=avg,color=sample, group=sample)) +
        geom_errorbar(aes(ymin=avg-std, ymax=avg+std)) +
        geom_point(aes(x=time, y=avg, color=sample)) +
        scale_y_continuous(expand=c(0,0), limits=c(0, 1.0), breaks=seq(0,1.0, by=0.2))

plot + ggtitle('Growth rates') + 
    xlab('Time (Hours)') +
    ylab('OD600') +
    theme(#axis.ticks.x = element_blank(),
          axis.text.x = element_text(color = 'black', size=12),
          axis.text.y = element_text(color = 'black', size=12),
          axis.title.x = element_text(size=16, hjust = 0.5, margin = margin(t=20)),
          axis.title.y = element_text(size=16, margin = margin(r=20)),
          axis.line.x = element_line(),
          axis.line.y = element_line(),
          plot.title = element_text(hjust = 0.5),
          panel.background = element_blank()
         ) +
    scale_color_brewer(palette='Set1')

