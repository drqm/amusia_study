working.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(working.dir); setwd('../../'); getwd()

data = read.csv('misc/subjects_info.csv',header = TRUE,sep=',',dec = ',')
data$age <- as.numeric(data$age)
data$MBEA <- as.numeric(data$Mean.MBEA)
data$MBEA_pitch <- as.numeric(data$Mean.MBEA.Pitch)

dem <- data.frame('item' = c('sample size','female','right-handed',
                             'education (years)','music training (years)',
                             'age', 'MBEA', 'MBEA pitch','PDT (semitones)')) 

for (g in unique(data$group) ) {
  
  dem[dem$item == 'sample size',g] <- sum(data$group == g)
  dem[dem$item == 'female',g] <- sum( (data$group == g) & 
                                               (data$sex == 'f') )
  dem[dem$item == 'right-handed',g] <- sum( (data$group == g) & 
                                                     (data$handedness == 'right') )
  dem[dem$item == 'education (years)',g] <- paste0(
    
    as.character(round( mean( data$academic.level[(data$group == g)] ),2)), ' (±',
    as.character(round( sd( data$academic.level[(data$group == g)] ),2)), ')'
    
  )
  
  dem[dem$item == 'music training (years)',g] <- paste0(
    
    as.character(round( mean( data$musicianship[(data$group == g)] ),2)), ' (±',
    as.character(round( sd( data$musicianship[(data$group == g)] ),2)), ')'
    
  )    
  dem[dem$item == 'age',g] <- paste0(
    
    as.character(round( mean( data$age[(data$group == g)] ),2)), ' (±',
    as.character(round( sd( data$age[(data$group == g)] ),2)), ')'
    
  )
  
  dem[dem$item == 'MBEA',g] <- paste0(
    
    as.character(round( mean( data$Mean.MBEA[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$Mean.MBEA[(data$group == g)] ),2)), ')'
    
  )
  
  dem[dem$item == 'MBEA pitch',g] <- paste0(
    
    as.character(round( mean( data$Mean.MBEA.Pitch[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$Mean.MBEA.Pitch[(data$group == g)] ),2)),')'
    
  )
  dem[dem$item == 'PDT (semitones)',g] <- paste0(
    
    as.character(round( mean( data$PDT[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$PDT[(data$group == g)] ),2)),')'
    
  )
}
colnames(dem)[1] <- c(' ')

dem
write.csv(dem, file = 'study1/results/demographics.csv',row.names = FALSE)

t.test(academic.level~group,data= data, paired = FALSE)
wilcox.test(musicianship~group,data= data)
t.test(age~group,data= data, paired = FALSE)
t.test(PDT~group,data= data, paired = FALSE)
t.test(Mean.MBEA~group,data= data, paired = FALSE)
t.test(Mean.MBEA.Pitch~group,data= data, paired = FALSE)

## Supplementary demographics:

dem2 <- data.frame('item' = c('MBEA scale', 'MBEA contour','MBEA interval', 'MBEA rhythm',
                              'MBEA meter', 'MBEA memory')) 

for (g in unique(data$group) ) {
  
  dem2[dem2$item == 'MBEA scale',g] <- paste0(
    
    as.character(round( mean( data$MBEA.scale[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$MBEA.scale[(data$group == g)] ),2)), ')'
    
  )
  
  dem2[dem2$item == 'MBEA contour',g] <- paste0(
    
    as.character(round( mean( data$MBEA.contour[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$MBEA.contour[(data$group == g)] ),2)), ')'
    
  )
  dem2[dem2$item == 'MBEA interval',g] <- paste0(
    
    as.character(round( mean( data$MBEA.interval[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$MBEA.interval[(data$group == g)] ),2)), ')'
    
  )
  dem2[dem2$item == 'MBEA rhythm',g] <- paste0(
    
    as.character(round( mean( data$MBEA.rhythm[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$MBEA.rhythm[(data$group == g)] ),2)), ')'
    
  )
  dem2[dem2$item == 'MBEA meter',g] <- paste0(
    
    as.character(round( mean( data$MBEA.meter[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$MBEA.meter[(data$group == g)] ),2)), ')'
    
  )
  dem2[dem2$item == 'MBEA memory',g] <- paste0(
    
    as.character(round( mean( data$MBEA.memory[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$MBEA.memory[(data$group == g)] ),2)), ')'
    
  )
}
colnames(dem2)[1] <- c(' ')

dem2
write.csv(dem2, file = 'study1/results/suppl1_MBEA_sub.csv',row.names = FALSE)