install.packages("terra")
library(terra)
library(ggplot2)
library(tidyverse)
library(caret)
library(doParallel)

r <- rast("/Users/lucas/Desktop/Stat Learning /NN Ecoregion Prediction/Thompson_and_others_2023_Atlas_data_release_netCDF_files/Bailey_ecoregions_on_25km_grid.nc")
names(r)

climate<- rast("/Users/lucas/Desktop/Stat Learning /NN Ecoregion Prediction/Thompson_and_others_2023_Atlas_data_release_netCDF_files/presence_absence_of_individual_species_on_25km_grid.nc")
names(climate)

#Variables I care about- 
#ANNT = mean annual temperature (degrees C) 17
#TMAX = absolute maximum temperature (degrees C)
#TMIN = absolute minimum temperature (degrees C)
#MTCO = mean temperature of the coldest month (degrees C)
#MTWA = mean temperature of the warmest month (degrees C) 21
#ANNP = mean annual total precipitation (mm) 34
#GDD5 = growing degree days (on a 5 degrees C base) 35
#AE/PE = moisture index, calculated as actual evapotranspiration (mm) divided by potential evapotranspiration (mm) 36


ANNT<-climate[[17]]
ANNT <- as.data.frame(ANNT, xy = TRUE, cells = TRUE, na.rm = TRUE)

TMAX<-climate[[18]]
TMAX <- as.data.frame(TMAX, xy = TRUE, cells = TRUE, na.rm = TRUE)

TMIN<-climate[[19]]
TMIN <- as.data.frame(TMIN, xy = TRUE, cells = TRUE, na.rm = TRUE)

MTCO<-climate[[20]]
MTCO <- as.data.frame(MTCO, xy = TRUE, cells = TRUE, na.rm = TRUE)

MTWA<-climate[[21]]
MTWA <- as.data.frame(MTWA, xy = TRUE, cells = TRUE, na.rm = TRUE)

ANNP<-climate[[34]]
ANNP <- as.data.frame(ANNP, xy = TRUE, cells = TRUE, na.rm = TRUE)

GDD5<-climate[[35]]
GDD5 <- as.data.frame(GDD5, xy = TRUE, cells = TRUE, na.rm = TRUE)

AE_PE<-climate[[36]]
AE_PE <- as.data.frame(AE_PE, xy = TRUE, cells = TRUE, na.rm = TRUE)

elevation<- r[[4]]
elevation <- as.data.frame(elevation, xy = TRUE, cells = TRUE, na.rm = TRUE)

all_regions <- r[[57]]
ecoregions <- as.data.frame(all_regions, xy = TRUE, cells = TRUE, na.rm = TRUE)


ecoregions<- left_join(ecoregions, ANNT, by= c("cell", "x", "y"))
ecoregions<- left_join(ecoregions, TMAX, by= c("cell", "x", "y"))
ecoregions<- left_join(ecoregions, TMIN, by= c("cell", "x", "y"))
ecoregions<- left_join(ecoregions, MTCO, by= c("cell", "x", "y"))
ecoregions<- left_join(ecoregions, MTWA, by= c("cell", "x", "y"))
ecoregions<- left_join(ecoregions, ANNP, by= c("cell", "x", "y"))
ecoregions<- left_join(ecoregions, GDD5, by= c("cell", "x", "y"))
ecoregions<- left_join(ecoregions, AE_PE, by= c("cell", "x", "y"))
ecoregions<- left_join(ecoregions, elevation, by= c("cell", "x", "y"))



northern<- ecoregions|>
  filter(ALL_BAILEY_ECOREGIONS < 13)

ggplot(northern) +
  geom_raster(aes(x = x, y = y, fill = `ALL_BAILEY_ECOREGIONS`)) +
  coord_equal() +
  scale_fill_viridis_c(name = "Ecoregion ID") +
  theme_minimal() +
  labs(title = "Predicted Ecoregion with Uncertainty")

northern$ALL_BAILEY_ECOREGIONS <- as.factor(make.names(northern$ALL_BAILEY_ECOREGIONS))
ecoregions$ALL_BAILEY_ECOREGIONS <- as.factor(make.names(ecoregions$ALL_BAILEY_ECOREGIONS))

# mapping variables I care about ------------------------------------------

ggplot(northern) +
  geom_raster(aes(x = x, y = y, fill = TMAX)) +
  coord_equal() +
  scale_fill_gradientn(
    colors = c("#FFF5EB", "#FDBE85", "#E6550D", "#990000"),
    name = "\u00B0C")+
  theme_minimal() +
  labs(title = "Maximum Temperature (\u00B0C)", x = "lat",
       y = "long")

ggplot(northern) +
  geom_raster(aes(x = x, y = y, fill = TMIN)) +
  coord_equal() +
  scale_fill_gradientn(
    colors = c("#DEEBF7", "#9ECAE1", "#3182BD", "#08519C"),
    name = "\u00B0C")+
  theme_minimal() +
  labs(title = "Minimum Temperature (\u00B0C)", x = "lat",
       y = "long")

ggplot(northern) +
  geom_raster(aes(x = x, y = y, fill = ANNP)) +
  coord_equal() +
  scale_fill_gradientn(
    colors = c("#F2F0F7", "#CBC9E2", "#9EBCDA", "#A6DBA0", "#41AB5D"),
    name = "Precipitation (mm)")+
  theme_minimal() +
  labs(title = "Mean Annual Precipitation (mm)", x = "lat",
       y = "long")

northern$elevation_capped <- pmin(northern$elevation, 2000)

ggplot(northern) +
  geom_raster(aes(x = x, y = y, fill = elevation)) +
  coord_equal() +
  scale_fill_gradientn(
    colors = c("#006837", "#78C679", "#FEE08B", "#D95F0E", "#A50026", "#FFFFFF"),
    limits = c(min(northern$elevation), 2000),
    oob = scales::squish,
    breaks = c(0, 500, 1000, 1500, 2000),
    labels = c("0", "500", "1000",  "1500", ">2000"),
    name = "Elevation (m)"
  ) +
  theme_minimal() +
  labs(title = "Elevation (m)", x = "lat", y = "long")

# NN ----------------------------------------------------------------------
inTrain <- createDataPartition(northern$ALL_BAILEY_ECOREGIONS, p = 3/4)[[1]] #creating a training testing dataset that maintains the even partitioning of classes
training <- northern[inTrain,]
testing <- northern[-inTrain,]

registerDoParallel(cores = 2)
numFolds <- trainControl(
  method = 'cv',
  number = 10,
  classProbs = TRUE,
  verboseIter = TRUE
)

tuneGrid = expand.grid(
  size = c(5, 10, 15, 20),      
  decay = c(0, 0.1, 0.5, 1e-3)   
) #testing the optimal node number and weights 

fit1 <- train(`ALL_BAILEY_ECOREGIONS` ~ . -ALL_BAILEY_ECOREGIONS -cell -x -y, data = training, method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid= tuneGrid)

results1 <- predict(fit1, newdata=testing)
conf1 <- confusionMatrix(results1, testing$ALL_BAILEY_ECOREGIONS)
conf1

plot(fit1) #the best fitting weight/node combo was 20 nodes and a weight decay of 0.001
# rf for variable importance ----------------------------------------------

rf <- train(
  ALL_BAILEY_ECOREGIONS ~ . - cell - x - y,
  data = training,
  method = "ranger", importance= "impurity"
)

varImp(rf)



# results -----------------------------------------------------------------


predicted_probs <- predict(fit2, newdata = northern, type = "prob") #soft prediction -- allows us to show the degree of uncertainty in areas


names<- c(1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9)
colnames(predicted_probs)<- names


predicted_probs$max<- max.col(predicted_probs)
predicted_probs$max_label <- colnames(predicted_probs)[max.col(predicted_probs)]
predicted_probs$opacity <- mapply(function(row, colname) predicted_probs[row, colname],
                                  row = seq_len(nrow(predicted_probs)),
                                  colname = predicted_probs$max_label)
predicted_probs<- bind_cols(predicted_probs, northern[, 1:3])
predicted_probs$max_label <- factor(predicted_probs$max_label, 
                                    levels = sort(as.numeric(unique(predicted_probs$max_label))))

ggplot(predicted_probs) +
  geom_raster(aes(x = x, y = y, fill = `max_label`, alpha = opacity)) +
  coord_equal() +
  scale_fill_viridis_d(name = "Ecoregion ID") +
  theme_minimal() +
  labs(title = "Predicted Ecoregion with Uncertainty")


ggplot(predicted_probs) +
  geom_raster(aes(x = x, y = y, fill = `opacity`)) +
  coord_equal() +
  scale_fill_gradientn(
    colors = c("#990000", "#CC3333", "#FF6666", "#FF9999", "#FFD1D1"),
    name = "Prediction Confidence")+
  theme_minimal() +
  labs(title = "Degree of Classification Uncertainty", x = "lat",
       y = "long")
 

# all ecoregions ----------------------------------------------------------

inTrain2 <- createDataPartition(ecoregions$ALL_BAILEY_ECOREGIONS, p = 3/4)[[1]] #creating a training testing dataset that maintains the even partitioning of classes
training2 <- ecoregions[inTrain2,]
testing2 <- ecoregions[-inTrain2,]

fit2 <- train(`ALL_BAILEY_ECOREGIONS` ~ . -ALL_BAILEY_ECOREGIONS -cell -x -y, data = training2, method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=expand.grid(size=c(10), decay=c(0.1)))

fit2
results2 <- predict(fit2, newdata=testing2)
conf2 <- confusionMatrix(results2, testing2$ALL_BAILEY_ECOREGIONS)
conf2

predicted_probs2 <- predict(fit2, newdata = ecoregions, type = "prob") #soft prediction -- allows us to show the degree of uncertainty in areas

predicted_probs2$max_label <- colnames(predicted_probs2)[max.col(predicted_probs2)]

predicted_probs2$opacity <- mapply(
  function(row, colname) predicted_probs2[row, colname],
  row = seq_len(nrow(predicted_probs2)),
  colname = predicted_probs2$max_label
)
predicted_probs2$max_label <- sub("^X", "", predicted_probs2$max_label)


predicted_probs2<- bind_cols(predicted_probs2, ecoregions[, 1:3])
predicted_probs2$max_label <- factor(predicted_probs2$max_label, 
                                    levels = sort(as.numeric(unique(predicted_probs2$max_label))))

ggplot(predicted_probs2) +
  geom_raster(aes(x = x, y = y, fill = `max_label`, alpha = opacity)) +
  coord_equal() +
  scale_fill_viridis_d(name = "Ecoregion ID") +
  theme_minimal() +
  labs(title = "Predicted Ecoregion with Uncertainty")


ggplot(predicted_probs2) +
  geom_raster(aes(x = x, y = y, fill = `opacity`)) +
  coord_equal() +
  scale_fill_gradientn(
    colors = c("#990000", "#CC3333", "#FF6666", "#FF9999", "#FFD1D1"),
    name = "Prediction Confidence")+
  theme_minimal() +
  labs(title = "Degree of Classification Uncertainty", x = "lat",
       y = "long")


# changing climate --------------------------------------------------------

climate_change<- ecoregions
climate_change$TMIN<- climate_change$TMIN + 20
climate_change$TMAX<- climate_change$TMAX + 10
climate_change$ANNT<- climate_change$ANNT + 7
climate_change$MTCO<- climate_change$MTCO + 10
climate_change$MTWA<- climate_change$MTWA + 10




change_preds <- predict(fit2, newdata = climate_change, type = "prob") #soft prediction -- allows us to show the degree of uncertainty in areas

change_preds$max_label <- colnames(change_preds)[max.col(change_preds)]

change_preds$opacity <- mapply(
  function(row, colname) change_preds[row, colname],
  row = seq_len(nrow(change_preds)),
  colname = change_preds$max_label
)
change_preds$max_label <- sub("^X", "", change_preds$max_label)


change_preds<- bind_cols(change_preds, climate_change[, 1:3])
change_preds$max_label <- factor(change_preds$max_label, 
                                     levels = sort(as.numeric(unique(change_preds$max_label))))

ggplot(change_preds) +
  geom_raster(aes(x = x, y = y, fill = `max_label`, alpha = opacity)) +
  coord_equal() +
  scale_fill_viridis_d(name = "Ecoregion ID") +
  theme_minimal() +
  labs(title = "Predicted Ecoregion with Uncertainty")

# individual species ------------------------------------------------------

species <- rast("/Users/lucas/Desktop/Stat Learning /Thompson_and_others_2023_Atlas_data_release_netCDF_files/presence_absence_of_individual_species_on_25km_grid.nc")
sugar <- species[[32]]
sugar_pa <- as.data.frame(sugar, xy = TRUE, cells = TRUE, na.rm = TRUE)


sugar <- ecoregions |>
  select(-ALL_BAILEY_ECOREGIONS)

sugar<- left_join(sugar, sugar_pa, by= c("cell","x","y"))

ggplot(sugar) +
  geom_raster(aes(x = x, y = y, fill = factor(Acer_saccharum))) +
  coord_equal() +
  scale_fill_manual(
    values = c("0" = "gray90", "1" = "#228B22"),
    name = "Presence",
    labels = c("Absent", "Present")
  ) +
  theme_minimal() +
  labs(title = "Presence of Acer saccharum", x = "lat", y = "long")

inTrain3 <- createDataPartition(sugar$Acer_saccharum, p = 3/4)[[1]] #creating a training testing dataset that maintains the even partitioning of classes
training3 <- sugar[inTrain3,]
testing3 <- sugar[-inTrain3,]

training3$Acer_saccharum <- factor(training3$Acer_saccharum, levels = c(0, 1), labels = c("absent", "present"))

fit3 <- train(`Acer_saccharum` ~ . -Acer_saccharum -cell -x -y, data = training3, method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=expand.grid(size=c(10), decay=c(0.1)))

results3 <- predict(fit3, newdata=testing3, type = "prob")
conf3 <- confusionMatrix(results3, testing3$Acer_saccharum)
conf3

pred_sugar_maple <- predict(fit3, newdata=sugar, type = "prob")

pred_sugar_maple$max_label <- colnames(pred_sugar_maple)[max.col(pred_sugar_maple)]

pred_sugar_maple$opacity <- mapply(
  function(row, colname) pred_sugar_maple[row, colname],
  row = seq_len(nrow(pred_sugar_maple)),
  colname = pred_sugar_maple$max_label
)

pred_sugar_maple<- bind_cols(pred_sugar_maple, sugar[, 1:3])

ggplot(pred_sugar_maple) +
  geom_raster(aes(x = x, y = y, fill = `max_label`)) +
  coord_equal() +
  scale_fill_viridis_d(name = "Ecoregion ID") +
  theme_minimal() +
  labs(title = "Predicted Ecoregion with Uncertainty")

maple_change<- predict(fit3, climate_change, type= "prob")
maple_change$max_label <- colnames(maple_change)[max.col(maple_change)]

maple_change$opacity <- mapply(
  function(row, colname) maple_change[row, colname],
  row = seq_len(nrow(maple_change)),
  colname = maple_change$max_label
)

maple_change<- bind_cols(maple_change, sugar[, 1:3])

ggplot(maple_change) +
  geom_raster(aes(x = x, y = y, fill = `max_label`)) +
  coord_equal() +
  scale_fill_viridis_d(name = "Ecoregion ID") +
  theme_minimal() +
  labs(title = "Predicted Ecoregion with Uncertainty")
#super strange results- you can tell that this doesn't nessecarily have all the variables it needs, like soil type. 



# white pine  -------------------------------------------------------------

pine <- species[[412]]
pine_pa <- as.data.frame(pine, xy = TRUE, cells = TRUE, na.rm = TRUE)


pine <- ecoregions |>
  select(-ALL_BAILEY_ECOREGIONS)

pine<- left_join(pine, pine_pa, by= c("cell","x","y"))

ggplot(pine) +
  geom_raster(aes(x = x, y = y, fill = factor(Picea_mariana))) +
  coord_equal() +
  scale_fill_manual(
    values = c("0" = "gray90", "1" = "#228B22"),
    name = "Presence",
    labels = c("Absent", "Present")
  ) +
  theme_minimal() +
  labs(title = "Presence of Picea mariana", x = "lat", y = "long")

inTrain4 <- createDataPartition(pine$Picea_mariana, p = 3/4)[[1]] #creating a training testing dataset that maintains the even partitioning of classes
training4 <- pine[inTrain4,]
testing4 <- pine[-inTrain4,]

training4$Picea_mariana <- factor(training4$Picea_mariana, levels = c(0, 1), labels = c("absent", "present"))

fit4 <- train(`Picea_mariana` ~ . -Picea_mariana -cell -x -y, data = training4, method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=expand.grid(size=c(10), decay=c(0.1)))

results4 <- predict(fit4, newdata=testing4, type = "prob")
conf4 <- confusionMatrix(results4, testing4$Picea_mariana)
conf4

pred_pine <- predict(fit4, newdata=pine, type = "prob")

pred_pine$max_label <- colnames(pred_pine)[max.col(pred_pine)]

pred_pine$opacity <- mapply(
  function(row, colname) pred_pine[row, colname],
  row = seq_len(nrow(pred_pine)),
  colname = pred_pine$max_label
)

pred_pine<- bind_cols(pred_pine, sugar[, 1:3])

ggplot(pred_pine) +
  geom_raster(aes(x = x, y = y, fill = `max_label`)) +
  coord_equal() +
  scale_fill_viridis_d(name = "Ecoregion ID") +
  theme_minimal() +
  labs(title = "Predicted Ecoregion with Uncertainty")


# cc spruce ---------------------------------------------------------------


spruce_change<- predict(fit4, climate_change, type= "prob")
spruce_change$max_label <- colnames(spruce_change)[max.col(spruce_change)]

spruce_change$opacity <- mapply(
  function(row, colname) spruce_change[row, colname],
  row = seq_len(nrow(spruce_change)),
  colname = spruce_change$max_label
)

spruce_change<- bind_cols(spruce_change, pine[, 1:3])

ggplot(spruce_change) +
  geom_raster(aes(x = x, y = y, fill = `max_label`)) +
  coord_equal() +
  scale_fill_viridis_d(name = "Ecoregion ID") +
  theme_minimal() +
  labs(title = "Predicted Ecoregion with Uncertainty")
#super strange results- you can tell that this doesn't nessecarily have all the variables it needs, like soil type. 







