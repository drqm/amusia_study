working.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(working.dir); setwd('../../'); getwd()

data = read.csv('misc/subjects_info.csv',header = TRUE,sep=';',dec = ',')
data$age <- as.numeric(data$age)
data$MBEA <- as.numeric(data$Mean.MBEA)
data$MBEA_pitch <- as.numeric(data$Mean.MBEA.Pitch)

dem <- data.frame('item' = c('sample size','female','right-handed',
                             'education (years)','music training (years)',
                             'age', 'MBEA', 'MBEA pitch','PDT (semitones)')) 

for (g in unique(data$group) ) {
  
  dem[which(dem$item == 'sample size'),g] <- sum(data$group == g)
  dem[which(dem$item == 'female'),g] <- sum( (data$group == g) & 
                                               (data$sex == 'f') )
  dem[which(dem$item == 'right-handed'),g] <- sum( (data$group == g) & 
                                                     (data$handedness == 'right') )
  dem[which(dem$item == 'education (years)'),g] <- paste0(
    
    as.character(round( mean( data$academic.level[(data$group == g)] ),2)), ' (±',
    as.character(round( sd( data$academic.level[(data$group == g)] ),2)), ')'
    
  )
  
  dem[which(dem$item == 'music training (years)'),g] <- paste0(
    
    as.character(round( mean( data$musicianship[(data$group == g)] ),2)), ' (±',
    as.character(round( sd( data$musicianship[(data$group == g)] ),2)), ')'
    
  )    
  dem[which(dem$item == 'age'),g] <- paste0(
    
    as.character(round( mean( data$age[(data$group == g)] ),2)), ' (±',
    as.character(round( sd( data$age[(data$group == g)] ),2)), ')'
    
  )
  
  dem[which(dem$item == 'MBEA'),g] <- paste0(
    
    as.character(round( mean( data$Mean.MBEA[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$Mean.MBEA[(data$group == g)] ),2)), ')'
    
  )
  
  dem[which(dem$item == 'MBEA pitch'),g] <- paste0(
    
    as.character(round( mean( data$Mean.MBEA.Pitch[(data$group == g)] ),2)),' (±',
    as.character(round( sd( data$Mean.MBEA.Pitch[(data$group == g)] ),2)),')'
    
  )
  dem[which(dem$item == 'PDT (semitones)'),g] <- paste0(
    
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
