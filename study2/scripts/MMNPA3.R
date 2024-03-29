#setwd('/Users/jonathannasielski/Desktop/UCU/S6/Thesis/amusia_study/')
setwd('C:/Users/au571303/Documents/projects/amusia_study/')
#install.packages("gridExtra")
library(ggplot2)
library(lme4)
library(ez)
library(emmeans)
library(reshape2)
library(gridExtra)

#Data set for all
d <- read.csv("data/MA/MA_toneness.csv", header = TRUE, sep = ",")
d <- d[d$feature!="pitch",]
d$MMN_amplitude <- d$MMN_amplitude/1e-6

#Data set for MMN
d1 <- melt(d, measure.vars = c("MMN_standard", "MMN_deviant"),
              id.vars = c("subject","feature","condition","group"))
colnames(d1)[colnames(d1)=="value"] <- "amplitude" 
colnames(d1)[colnames(d1)=="variable"] <- "deviance"
d1$amplitude <- d1$amplitude/1e-6

#Data set for P3a
d2 <- melt(d, measure.vars = c("P3a_standard", "P3a_deviant"),
           id.vars = c("subject","feature","condition","group"))
colnames(d2)[colnames(d2)=="value"] <- "amplitude" 
colnames(d2)[colnames(d2)=="variable"] <- "deviance"
d2$amplitude <- d2$amplitude/1e-6

#check the MMN
s1 <- lmer(amplitude ~deviance*condition*feature*group+(1|subject),data = d1)
confint.MMNcheck <- as.data.frame(confint(lsmeans(s1,pairwise~deviance|condition|feature|group
                                              )$contrast))
pairwise.MMNcheck <-cbind(as.data.frame(lsmeans(s1, pairwise~deviance|condition|feature|group
                                            )$contrasts),
                      confint.MMNcheck[c("lower.CL","upper.CL")])
pairwise.MMNcheck <- pairwise.MMNcheck[c("contrast","condition","feature","group","estimate","lower.CL",
                                 "upper.CL","t.ratio","p.value")]
pairwise.MMNcheck[,5:8] <- round(pairwise.MMNcheck[,5:8],2)
pairwise.MMNcheck[9] <- round(pairwise.MMNcheck[9],6)
colnames(pairwise.MMNcheck) <- c("contrast","condition","feature","group","estimate",
                             "CI 2.5%","CI 97.5%","t","p")

write.table(pairwise.MMNcheck,file="study2/results/pairwise_MMNcheck.csv",
            sep=";",row.names = FALSE,quote=FALSE) # Export to a table

#check the P3a
s2 <- lmer(amplitude ~deviance*condition*feature*group+(1|subject),data = d2)
confint.P3acheck <- as.data.frame(confint(lsmeans(s2,pairwise~deviance|condition|feature|group)$contrast))
pairwise.P3acheck <-cbind(as.data.frame(lsmeans(s2, pairwise~deviance|condition|feature|group)$contrasts),
                      confint.P3acheck[c("lower.CL","upper.CL")])
pairwise.P3acheck <- pairwise.P3acheck[c("contrast","condition","feature","group","estimate","lower.CL",
                                         "upper.CL","t.ratio","p.value")]
pairwise.P3acheck[,5:8] <- round(pairwise.P3acheck[,5:8],2)
pairwise.P3acheck[9] <- round(pairwise.P3acheck[9],6)
colnames(pairwise.P3acheck) <- c("contrast","condition","feature","group","estimate",
                                 "CI 2.5%","CI 97.5%","t","p")

write.table(pairwise.P3acheck,file="study2/results/pairwise_P3acheck.csv",
            sep=";",row.names = FALSE,quote=FALSE) # Export to a table

### Extract MMN
#Amplitude mean and anova

#ggplot(d,aes(x=condition, y=MMN_amplitude, color=group))+
  #geom_jitter(width = 0.1)+
  #facet_wrap(~feature)

m1 <- lmer(MMN_amplitude ~condition*feature*group+(1|subject),data = d)

aov1 <- ezANOVA(data= d, MMN_amplitude,subject,within = c(condition,feature),
                between = c(group),detailed = TRUE)

#pairwise comparaison for Post hoc amplitude
confint.MMNamp <- as.data.frame(confint(lsmeans(m1,pairwise~condition|feature,
                                              adjust = "Bonferroni")$contrast))
pairwise.MMNamp <-cbind(as.data.frame(lsmeans(m1, pairwise~condition|feature,
                                            adjust="bonferroni")$contrasts),
                      confint.MMNamp[c("lower.CL","upper.CL")])

pairwise.MMNamp[,"d"] <- pairwise.MMNamp$estimate/sqrt(VarCorr(m1)$subject[1] + sigma(m1)^2)
pairwise.MMNamp <- pairwise.MMNamp[c("feature","contrast","estimate","lower.CL",
                                 "upper.CL","t.ratio","d","p.value")]
pairwise.MMNamp[,3:7] <- round(pairwise.MMNamp[,3:7],2)
pairwise.MMNamp[8] <- round(pairwise.MMNamp[8],6)
colnames(pairwise.MMNamp) <- c("feature","contrast","estimate",
                             "CI 2.5%","CI 97.5%","t","d","p")

write.table(pairwise.MMNamp,file="study2/results/pairwise_MMNamp.csv",
            sep=";",row.names = FALSE,quote=FALSE) # Export to a table

### Pairwise 2 + group
confint.MMNamp <- as.data.frame(confint(lsmeans(m1,pairwise~condition|group|feature,
                                                adjust = "Bonferroni")$contrast))
pairwise.MMNamp <-cbind(as.data.frame(lsmeans(m1, pairwise~condition|group|feature,
                                              adjust="bonferroni")$contrasts),
                        confint.MMNamp[c("lower.CL","upper.CL")])

pairwise.MMNamp[,"d"] <- pairwise.MMNamp$estimate/sqrt(VarCorr(m1)$subject[1] + sigma(m1)^2)
pairwise.MMNamp <- pairwise.MMNamp[c("feature","group","contrast","estimate","lower.CL",
                                     "upper.CL","t.ratio","d","p.value")]
pairwise.MMNamp[,4:8] <- round(pairwise.MMNamp[,4:8],2)
pairwise.MMNamp[9] <- round(pairwise.MMNamp[9],6)
colnames(pairwise.MMNamp) <- c("feature","group","contrast","estimate",
                               "CI 2.5%","CI 97.5%","t","d","p")

write.table(pairwise.MMNamp,file="study2/results/pairwise_MMNamp1.csv",
            sep=";",row.names = FALSE,quote=FALSE) # Export to a table

#plot MMN Amplitude
MMNamp <- ggplot(d,aes(condition, MMN_amplitude)) +
  geom_hline(yintercept = 0) +
  geom_point(alpha = 0.6,color = 'black',size = 0.8) +
  geom_line(aes(group = subject), alpha = 0.15) +
  geom_boxplot(aes(fill = group),alpha = 0.3,
               color = 'black', width = 0.2) +
  geom_violin(aes(color = group),color = 'black',
              alpha = 0.2,trim = FALSE) + 
  scale_fill_manual(values = c('blue','red')) +
  facet_grid(feature~group) +
  xlab('conditions') +
  ylab('mean amplitude (μV)') +
  scale_x_discrete(labels=c("hihat", "optimal",
                            "hihat", "optimal")) +
  theme_bw() +
  theme(legend.position = "none"); MMNamp

#Latency mean and anova 

#ggplot(d,aes(x=condition, y=MMN_latency, color=group))+
  #geom_jitter(width = 0.1)+
  #facet_wrap(~feature)

m2 <- lmer(MMN_latency ~condition*feature*group+(1|subject),data = d)

aov2 <- ezANOVA(data= d, MMN_latency,subject,within = c(condition,feature),
                between = c(group),detailed = TRUE)

#pairwise comparaison for Post hoc latency
confint.MMNlat <- as.data.frame(confint(lsmeans(m2,pairwise~condition|feature,
                                               adjust = "Bonferroni")$contrast))
pairwise.MMNlat <-cbind(as.data.frame(lsmeans(m2, pairwise~condition|feature,
                                             adjust="bonferroni")$contrasts),
                       confint.MMNlat[c("lower.CL","upper.CL")])

pairwise.MMNlat[,"d"] <- pairwise.MMNlat$estimate/sqrt(VarCorr(m2)$subject[1] + sigma(m2)^2)
pairwise.MMNlat <- pairwise.MMNlat[c("feature","contrast","estimate","lower.CL",
                                   "upper.CL","t.ratio","d","p.value")]
pairwise.MMNlat[,3:7] <- round(pairwise.MMNlat[,3:7],2)
pairwise.MMNlat[8] <- round(pairwise.MMNlat[8],6)
colnames(pairwise.MMNlat) <- c("feature","contrast","estimate",
                              "CI 2.5%","CI 97.5%","t","d","p")

write.table(pairwise.MMNlat,file="study2/results/pairwise_MMNlat.csv",
            sep=";",row.names = FALSE,quote=FALSE) # Export to a table

#plot MMN latencies
MMNlat <- ggplot(d,aes(condition, MMN_latency)) +
  geom_hline(yintercept = 0) +
  geom_point(alpha = 0.6,color = 'black',size = 0.8) +
  geom_line(aes(group = subject), alpha = 0.15) +
  geom_boxplot(aes(fill = group),alpha = 0.3,
               color = 'black', width = 0.2) +
  geom_violin(aes(color = group),color = 'black',
              alpha = 0.2,trim = FALSE) + 
  scale_fill_manual(values = c('blue','red')) +
  facet_grid(feature~group) +
  xlab('conditions') +
  ylab('mean latencies (ms)') +
  scale_x_discrete(labels=c("hihat", "optimal",
                            "hihat", "optimal")) +
  theme_bw() +
  theme(legend.position = "none"); MMNlat

#group plot together 
MMNplots <- arrangeGrob(MMNamp,MMNlat,ncol=2);plot(MMNplots)
ggsave("../amusia_study/study2/results/MMN_complexity.pdf", plot=MMNplots,width = 170, height = 150, units = 'mm', dpi = 300)
ggsave("../amusia_study/study2/results/MMN_complexity.png", plot=MMNplots,width = 170, height = 150, units = 'mm', dpi = 300)

### Extract P3a
#Amplitude mean and anova
d$P3a_amplitude <- d$P3a_amplitude/1e-6

#ggplot(d,aes(x=condition, y=P3a_amplitude, color=group))+
  #geom_jitter(width = 0.1)+
  #facet_wrap(~feature)

m3 <- lmer(P3a_amplitude ~condition*feature*group+(1|subject),data = d)

aov3 <- ezANOVA(data= d, P3a_amplitude,subject,within = c(condition,feature),
                between = c(group),detailed = TRUE)

#pairwise comparaison for Post hoc amplitude
confint.P3aamp <- as.data.frame(confint(lsmeans(m3,pairwise~condition|feature,
                                                adjust = "Bonferroni")$contrast))
pairwise.P3aamp <-cbind(as.data.frame(lsmeans(m3, pairwise~condition|feature,
                                              adjust="bonferroni")$contrasts),
                        confint.P3aamp[c("lower.CL","upper.CL")])

pairwise.P3aamp[,"d"] <- pairwise.P3aamp$estimate/sqrt(VarCorr(m3)$subject[1] + sigma(m3)^2)
pairwise.P3aamp <- pairwise.P3aamp[c("feature","contrast","estimate","lower.CL",
                                     "upper.CL","t.ratio","d","p.value")]
pairwise.P3aamp[,3:7] <- round(pairwise.P3aamp[,3:7],2)
pairwise.P3aamp[8] <- round(pairwise.P3aamp[8],6)
colnames(pairwise.P3aamp) <- c("feature","contrast","estimate",
                               "CI 2.5%","CI 97.5%","t","d","p")

write.table(pairwise.P3aamp,file="study2/results/pairwise_P3aamp.csv",
            sep=";",row.names = FALSE,quote=FALSE) # Export to a table

#plot P3a amplitudes
P3aamp <- ggplot(d,aes(condition, P3a_amplitude)) +
  geom_hline(yintercept = 0) +
  geom_point(alpha = 0.6,color = 'black',size = 0.8) +
  geom_line(aes(group = subject), alpha = 0.15) +
  geom_boxplot(aes(fill = group),alpha = 0.3,
               color = 'black', width = 0.2) +
  geom_violin(aes(color = group),color = 'black',
              alpha = 0.2,trim = FALSE) + 
  scale_fill_manual(values = c('blue','red')) +
  facet_grid(feature~group) +
  xlab('conditions') +
  ylab('mean amplitude (μV)') +
  scale_x_discrete(labels=c("hihat", "optimal",
                            "hihat", "optimal")) +
  theme_bw() +
  theme(legend.position = "none"); P3aamp

#Latency mean and anova 

#ggplot(d,aes(x=condition, y=P3a_latency, color=group))+
  #geom_jitter(width = 0.1)+
  #facet_wrap(~feature)

m4 <- lmer(P3a_latency ~condition*feature*group+(1|subject),data = d)

aov4 <- ezANOVA(data= d, P3a_latency,subject,within = c(condition,feature),
                between = c(group),detailed = TRUE)

#pairwise comparaison for Post hoc latency
confint.P3alat <- as.data.frame(confint(lsmeans(m4,pairwise~condition|feature,
                                               adjust = "Bonferroni")$contrast))
pairwise.P3alat <-cbind(as.data.frame(lsmeans(m4, pairwise~condition|feature,
                                             adjust="bonferroni")$contrasts),
                       confint.P3alat[c("lower.CL","upper.CL")])

pairwise.P3alat[,"d"] <- pairwise.P3alat$estimate/sqrt(VarCorr(m4)$subject[1] + sigma(m4)^2)
pairwise.P3alat <- pairwise.P3alat[c("feature","contrast","estimate","lower.CL",
                                   "upper.CL","t.ratio","d","p.value")]
pairwise.P3alat[,3:7] <- round(pairwise.P3alat[,3:7],2)
pairwise.P3alat[8] <- round(pairwise.P3alat[8],6)
colnames(pairwise.P3alat) <- c("feature","contrast","estimate",
                              "CI 2.5%","CI 97.5%","t","d","p")

write.table(pairwise.P3alat,file="study2/results/pairwise_P3alat.csv",
            sep=";",row.names = FALSE,quote=FALSE) # Export to a table

#pairwise comparaison for Post hoc latency with group
confint.P3alat1 <- as.data.frame(confint(lsmeans(m4,pairwise~condition|group|feature,
                                                adjust = "Bonferroni")$contrast))
pairwise.P3alat1 <-cbind(as.data.frame(lsmeans(m4, pairwise~condition|group|feature,
                                              adjust="bonferroni")$contrasts),
                        confint.P3alat1[c("lower.CL","upper.CL")])

pairwise.P3alat1[,"d"] <- pairwise.P3alat1$estimate/sqrt(VarCorr(m4)$subject[1] + sigma(m4)^2)
pairwise.P3alat1 <- pairwise.P3alat1[c("feature","group","contrast","estimate","lower.CL",
                                     "upper.CL","t.ratio","d","p.value")]
pairwise.P3alat1[,4:8] <- round(pairwise.P3alat1[,4:8],2)
pairwise.P3alat1[9] <- round(pairwise.P3alat1[9],6)
colnames(pairwise.P3alat1) <- c("feature","group","contrast","estimate",
                               "CI 2.5%","CI 97.5%","t","d","p")

write.table(pairwise.P3alat1,file="study2/results/pairwise_P3alat1.csv",
            sep=";",row.names = FALSE,quote=FALSE) # Export to a table

#pairwise comparaison for Post hoc latency without feature with group
confint.P3alat2 <- as.data.frame(confint(lsmeans(m4,pairwise~condition|group,
                                                adjust = "Bonferroni")$contrast))
pairwise.P3alat2 <-cbind(as.data.frame(lsmeans(m4, pairwise~condition|group,
                                              adjust="bonferroni")$contrasts),
                        confint.P3alat2[c("lower.CL","upper.CL")])

pairwise.P3alat2[,"d"] <- pairwise.P3alat2$estimate/sqrt(VarCorr(m4)$subject[1] + sigma(m4)^2)
pairwise.P3alat2 <- pairwise.P3alat2[c("group","contrast","estimate","lower.CL",
                                     "upper.CL","t.ratio","d","p.value")]
pairwise.P3alat2[,3:7] <- round(pairwise.P3alat2[,3:7],2)
pairwise.P3alat2[8] <- round(pairwise.P3alat2[8],6)
colnames(pairwise.P3alat2) <- c("group","contrast","estimate",
                               "CI 2.5%","CI 97.5%","t","d","p")

write.table(pairwise.P3alat2,file="study2/results/pairwise_P3alat2.csv",
            sep=";",row.names = FALSE,quote=FALSE) # Export to a table

#plot P3a latencies
P3alat <- ggplot(d,aes(condition, P3a_latency)) +
  geom_hline(yintercept = 0) +
  geom_point(alpha = 0.6,color = 'black',size = 0.8) +
  geom_line(aes(group = subject), alpha = 0.15) +
  geom_boxplot(aes(fill = group),alpha = 0.3,
               color = 'black', width = 0.2) +
  geom_violin(aes(color = group),color = 'black',
              alpha = 0.2,trim = FALSE) + 
  scale_fill_manual(values = c('blue','red')) +
  facet_grid(feature~group) +
  xlab('conditions') +
  ylab('mean latencies (ms)') +
  scale_x_discrete(labels=c("hihat", "optimal",
                            "hihat", "optimal")) +
  theme_bw() +
  theme(legend.position = "none"); P3alat

#group plot together 
P3aplots <- arrangeGrob(P3aamp,P3alat,ncol=2);plot(P3aplots)
ggsave("../amusia_study/study2/results/P3a_complexity.pdf", plot=P3aplots,width = 170, height = 150, units = 'mm', dpi = 300)
ggsave("../amusia_study/study2/results/P3a_complexity.png", plot=P3aplots,width = 170, height = 150, units = 'mm', dpi = 300)