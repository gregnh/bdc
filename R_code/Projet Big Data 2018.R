#Libraries
library(dummies)

#Working Directory

setwd("~/Desktop/data_meteo")

list_all_files <- list.files()
list_train_dataset <- list_all_files[-c(1,2)]

create_train_dataset <- function(filename){
  
  dataset <- read.csv(file = filename,header = T,sep = ';',dec = ',')
  
  #Month variable
  dataset<-dummy.data.frame(dataset, names = c("mois"),sep = '_')
  
  #Case were not all months are represented in the dataset
  if(length(colnames(dataset)) != 42){
    cat("There are ", 42 - length(colnames(dataset))," months missing in the file: ", filename," which are","\n")
    
    #Creation of the "0 columns" for the missing month
    for (i in c("mois_janvier","mois_février","mois_mars","mois_avril","mois_mai","mois_juin","mois_juillet","mois_août","mois_septembre","mois_octobre","mois_novembre","mois_décembre")){
      if (is.element(i, colnames(dataset)) == F){
        dataset[i] <- rep(0, nrow(dataset))
        print(substring(i,6))
      }
    }
    cat("\n")
    print("WARNING")
    cat("\n")
  }
  
  #Case when all month variables are created
  colnames(dataset)[31:42] <- sapply(colnames(dataset)[31:42], function(x) substring(x,6))
  dataset <- dataset[,c(colnames(dataset)[1:30],"janvier","février","mars","avril","mai","juin","juillet","août","septembre","octobre","novembre","décembre")]
  
  
  return(dataset)
}

#Creation of the large dataset

final_dataset <- function(){
  
  final_dataset <- create_train_dataset(list_train_dataset[1])
  cat("\n")
  print("Processing the files !")
  cat("\n")
  
  for (i in list_train_dataset[-c(1)]){
    cat("Processing the file: ",i," 'Done'", "\n")
    final_dataset <- rbind(final_dataset,create_train_dataset(i))
  }
  return(final_dataset)
}

#Final dataset
data<-final_dataset()

str(data)
class(data$flvis1SOL0) <- "numeric"
class(data$ddH10_rose4) <- "numeric"

#Substetting the dataset and order it !
subset_insee <- function(dataset,liste_insee){
  cat("Subsetting among the", paste(liste_insee,collapse = " and "),"town(s)", "\n")
  cat("\n")
  dataset <- subset(dataset,insee %in% liste_insee)
  
  #dataset <- dataset[,-c(which(colnames(dataset)=="insee"))] 
  
  dataset <- dataset[order(dataset$date, dataset$ech),]
  return(dataset)
}

Nice <- subset_insee(data,c(6088001))
Toulouse <- subset_insee(data,c(31069001))
Bordeaux <- subset_insee(data,c(33281001))
Rennes <- subset_insee(data,c(35281001))
Lille <- subset_insee(data,c(59343001))
Strasbourg <- subset_insee(data,c(67124001))
Paris <- subset_insee(data,c(75114001))


#Explication : cf Gregoire
cheating <- function(dataset){
  days <- unique(dataset$date)
  for (i in 1:length(days[-c(length(days))])){
    print(i)
    day1 <- subset(dataset,date == days[i] & ech >=25)
    day1 <- day1[order(day1$ech),]
    
    day_next <- subset(dataset,date == days[i+1] & ech >=1)
    day_next <- day_next[order(day_next$ech),]
    
    #Creation du "bon" dataset
    new_day <- cbind(day1[,c(1:3)],day_next[c(1:dim(day1)[1]),c(4:dim(day_next)[2])])
    new_day[,which(colnames(new_day)=="ech")] <- new_day[,which(colnames(new_day)=="ech")] + 24
    
    #Changement du dataset original
    ID_row_to_chande <- which(dataset$date == days[i] & dataset$ech >=25)
    j <- 1
    for (i in ID_row_to_chande){
      dataset[i,] <- day1[j,]
      j <- j + 1
    }
  }
  return(dataset)
}

#Dataset using cheating
Nice_C <- cheating(Nice)
Toulouse_C <- cheating(Toulouse)
Bordeaux_C <- cheating(Bordeaux)
Rennes_C <- cheating(Rennes)
Lille_C <- cheating(Lille)
Strasbourg_C <- cheating(Strasbourg)
Paris_C <- cheating(Paris)




#Mise au carré/cube/cos pour certaines variables
new_variables <- function(dataset){
  resultat <- dataset[,c(1:3)]
  
  #ID variable to change
  names_variable <- colnames(dataset)[4:29]
  for (i in names_variable){
    new_col_name_2 <- paste0(i,"_sqr")
    new_col_name_3 <- paste0(i,"_cub")
    new_col_name_cos <- paste0(i,"_cos")
    
    resultat[,new_col_name_2] <- dataset[,i]**2
    resultat[,new_col_name_3] <- dataset[,i]**3
    resultat[,new_col_name_cos] <- cos(dataset[,i])
  }
  return(resultat)
}

#Final Datasets 
Nice_Final <- new_variables(Nice)
Toulouse_Final <- new_variables(Toulouse)
Bordeaux_Final <- new_variables(Bordeaux)
Rennes_Final <- new_variables(Rennes)
Lille_Final <- new_variables(Lille)
Strasbourg_Final <- new_variables(Strasbourg)
Paris_Final <- new_variables(Paris)