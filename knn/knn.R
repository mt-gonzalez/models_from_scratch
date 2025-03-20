# Distances Formulas
manhattan_distance <- function(x_1, x_2) {
  d <- sum(abs(x_1 - x_2))
  
  return(d)
}

euclidean_distance <- function(x_1, x_2) {
  d <- sum(sqrt((x_1 - x_2)^2))
  
  return(d)
}

cosine_distance <- function(x_1, x_2) {
  dot_product <- sum(x_1*x_2)
  
  norm_x1 <- sqrt(sum(x_1^2))
  norm_x2 <- sqrt(sum(x_2^2))
  
  similarity <- dot_product / (norm_x1 * norm_x2)
  d <- (1 - similarity)
  
  if (abs(d) < 1e-15) { # Managing numerical error
    return(0)
  }
  
  return(d)
}

# Class creation
create_model <- function(k=3) {
  knn_model = list(
    k = k,
    X_train = NULL,
    Y_train = NULL
  )
  
  class(knn_model) <- "KNN"
  return(knn_model)

}

fit.KNN <- function(model, X_train, Y_train) {
  model$X_Train <- X_train
  model$Y_train <- Y_train
  
  return(model)
}

gather_distances.KNN <- function(model, x_0, d_formula = euclidean_distance) {
  distances <- apply(model$X_train, 1, function(x_train) d_formula(x_0, x_train))
  
  return(distances)
}

getNeighbours.KNN <- function(model, distances) {
  sorted_indices <- order(distances)
  neighbours <- model$Y_train[sorted_indices[1:model$k]]
  
  return(neighbours)
}

most_common.KNN <- function(model, neighbours) {
  neighbours_count <- table(neighbours)
  most_common <- names(neighbours_count)[which.max(neighbours_count)]
  
  return(most_common)
}

predict.KNN <- function(model, x_0) {
  distances <- gather_distances.KNN(model, x_0)
  k_neighbours <- getNeighbours.KNN(model, distances)
  predicted_class <- most_common.KNN(model, k_neighbours)
  
  return(predicted_class)
}