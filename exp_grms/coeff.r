library(statnet)
library(ergm)
library(sna)


# parse_bias_value_to_number <- function(value) {
#   if (value == "rand") {
#     return(1)
#   } else {
#     return(as.numeric(value))
#   }
# }

parse_folder_name <- function(folder_name) {
  result <- c()

  stplt_names <- unlist(strsplit(folder_name, "_"))

  for (name in stplt_names) {
    bias <- unlist(strsplit(name, "-"))

    if (bias[1] == "sd" && bias[2] == "rand") {
      result["sp"] <- 1
      next
    }

    if (bias[1] == "sd" && bias[2] == "exp") next


    if (!is.element(bias[1], names(result))) {
      result[bias[1]] <- as.numeric(bias[2])
    }
  }

  return(result)
}

process_folder <- function(walk_dir) {
  name <- unlist(strsplit(walk_dir, "\\\\"))

  folder_name <- parse_folder_name(name[length(name)])
  target <- c(folder_name["r"], folder_name["sp"], folder_name["d"])


  members <- read.csv(paste0(walk_dir, "\\members.csv"))


  transactions <- read.csv(paste0(walk_dir, "\\transactions.csv"))

  g <- network(transactions,
    loops = TRUE, directed = TRUE,
    multiple = TRUE,
    vertex.attr = members,
  )

  deg <- degree(dat = g)
  bet <- betweenness(dat = g, diag = TRUE)
  cls <- closeness(dat = g, cmode = "suminvdir")
  transi <- gtrans(dat = g, diag = TRUE, use.adjacency = FALSE)
  ecc <- grecip(dat = g)
  eig <- evcent(dat = g, diag = TRUE, )
  deg_centr <- centralization(dat = g, FUN = degree, diag = TRUE)
  bet_centr <- centralization(dat = g, FUN = betweenness, diag = TRUE)
  cls_centr <- centralization(dat = g, FUN = closeness, diag = TRUE)

  set.vertex.attribute(g, "deg", deg)
  set.vertex.attribute(g, "bet", bet)
  set.vertex.attribute(g, "cls", cls)
  set.vertex.attribute(g, "transi", transi)
  set.vertex.attribute(g, "ecc", ecc)
  set.vertex.attribute(g, "eig", eig)
  set.vertex.attribute(g, "deg_centr", deg_centr)
  set.vertex.attribute(g, "bet_centr", bet_centr)
  set.vertex.attribute(g, "cls_centr", cls_centr)




  ergm <- ergm(
    formula = g ~
      nodematch("deg") +
      nodematch("bet") +
      nodematch("cls") +
      nodematch("transi") +
      nodematch("ecc") +
      nodematch("eig") +
      nodematch("deg_centr") +
      nodematch("bet_centr") +
      nodematch("cls_centr"),
    control = control.ergm()
  )

  coefficients <- coef(ergm)
  result <- c(coefficients, target, c(id = walk_dir))

  print(result)
}

process_folder("C:\\Users\\bugl2301\\Documents\\beluga\\test_report\\iter-1_r-0.8_sd-rand_sp-0.7000000000000001_d-0.9")
