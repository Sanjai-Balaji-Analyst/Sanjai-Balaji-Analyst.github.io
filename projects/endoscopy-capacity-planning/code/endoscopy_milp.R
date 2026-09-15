# ==============================================================================
# Semiannual Capacity Planning for an Endoscopy Unit — MILP
# Data Driven Decision Making, MSc Business Analytics, Queen's University Belfast
# Author: Sanjai Balaji (40478904)
#
# Allocates 10 procedure rooms across a 26-week horizon to diagnostic /
# therapeutic configurations, minimising total setup + operating cost subject
# to weekly demand, room capacity, and clinician-availability constraints.
# Formulated and solved as a Mixed-Integer Linear Program with ompr + GLPK.
# ==============================================================================

## install.packages(c("ompr","ompr.roi","ROI.plugin.glpk","dplyr","magrittr","ggplot2"))

library(ompr)
library(ompr.roi)
library(ROI.plugin.glpk)
library(dplyr)
library(magrittr)
library(ggplot2)

# ---- Room profiles -----------------------------------------------------------
room <- 1:10
room_type <- c("small", "small", "small", "medium", "medium", "medium", "medium",
               "large", "large", "large")

# Weekly procedure-hour capacities, by room
diag_capacity  <- c(60, 60, 60, 120, 120, 120, 120, 180, 180, 180)
thera_capacity <- c(30, 30, 30, 60, 60, 60, 60, 120, 120, 120)

# Costs, by room
setup_cost      <- c(15000, 15000, 15000, 20000, 20000, 20000, 20000, 25000, 25000, 25000)
allocation_cost <- c(180, 180, 180, 200, 200, 200, 200, 220, 220, 220)  # per hour

# Weekly demand and clinician availability, 26 weeks
P_diag <- c(230, 235, 225, 230, 240, 245, 280, 295, 300, 285, 240, 235, 230, 225, 230,
            235, 240, 245, 250, 245, 240, 235, 230, 230, 235, 230)   # diagnostic hours
P_thera <- c(145, 145, 150, 150, 155, 160, 168, 176, 182, 188, 190, 192, 188, 195, 187,
             182, 178, 172, 168, 165, 162, 158, 155, 152, 150, 148)  # therapeutic hours
C_w <- c(520, 515, 525, 520, 530, 525, 490, 480, 485, 500, 510, 515, 520, 525, 530, 525,
         520, 480, 475, 500, 510, 515, 520, 525, 530, 520)           # clinician hours
weeks <- 1:26

# ---- Building the model -------------------------------------------------------
model <- MIPModel() %>%
  # Decision variables
  add_variable(x_diag[r, w], r = room, w = weeks, type = "binary") %>%
  add_variable(x_thera[r, w], r = room, w = weeks, type = "binary") %>%
  add_variable(d[r, w], r = room, w = weeks, type = "continuous", lb = 0) %>%
  add_variable(t[r, w], r = room, w = weeks, type = "continuous", lb = 0) %>%
  add_variable(s[r, w], r = room, w = weeks, type = "binary")

# Room configuration: at most one procedure type per room per week
model <- model %>%
  add_constraint(x_diag[r, w] + x_thera[r, w] <= 1, r = room, w = weeks)

# Room capacity: hours scheduled cannot exceed the room's capacity, and only
# apply if the room is configured for that procedure type this week
model <- model %>%
  add_constraint(d[r, w] <= diag_capacity[r] * x_diag[r, w], r = room, w = weeks) %>%
  add_constraint(t[r, w] <= thera_capacity[r] * x_thera[r, w], r = room, w = weeks)

# Demand satisfaction: weekly diagnostic / therapeutic demand must be met
model <- model %>%
  add_constraint(sum_expr(d[r, w], r = room) >= P_diag[w], w = weeks) %>%
  add_constraint(sum_expr(t[r, w], r = room) >= P_thera[w], w = weeks)

# Clinician availability: total hours scheduled cannot exceed clinician capacity
model <- model %>%
  add_constraint(sum_expr(d[r, w] + t[r, w], r = room) <= C_w[w], w = weeks)

# Setup logic: a setup cost is incurred whenever a room becomes active,
# whether newly active in week 1 or changing from inactive to active later
model <- model %>%
  add_constraint(s[r, 1] >= x_diag[r, 1] + x_thera[r, 1], r = room)

model <- model %>%
  add_constraint(
    s[r, w] >= x_diag[r, w] + x_thera[r, w] - (x_diag[r, w - 1] + x_thera[r, w - 1]),
    r = room, w = 2:26
  )

# Objective: minimise total setup + operating cost across the horizon
model <- model %>%
  set_objective(sum_expr(setup_cost[r] * s[r, w] + allocation_cost[r] * (d[r, w] + t[r, w]),
                          r = room, w = weeks), sense = "min")

# ---- Solve ----------------------------------------------------------------------
# 10-minute time limit (600,000 ms); GLPK reports the best bound found if the
# limit is hit before proving optimality (see solver log in the report).
result <- solve_model(
  model,
  with_ROI(
    solver = "glpk",
    verbose = TRUE,
    control = list(tm_limit = 600000)
  )
)

solver_status(result)
objective_value(result)

get_solution(result, d[r, w])
get_solution(result, t[r, w])
get_solution(result, x_diag[r, w])
get_solution(result, x_thera[r, w])

# Sanity check against total requested demand
sum(P_diag)
sum(P_thera)
summary(get_solution(result, d[r, w])$value)

# ---- Visualisation ---------------------------------------------------------------
total_diag <- get_solution(result, d[r, w]) %>% group_by(w) %>% summarise(total = sum(value))
ggplot(total_diag, aes(x = w, y = total)) +
  geom_bar(stat = "identity") +
  labs(title = "Total Diagnostic Hours per Week")

total_thera <- get_solution(result, t[r, w]) %>% group_by(w) %>% summarise(total = sum(value))
ggplot(total_thera, aes(x = w, y = total)) +
  geom_bar(stat = "identity") +
  labs(title = "Total Therapeutic Hours per Week")

room_usage <- get_solution(result, d[r, w]) %>% group_by(r) %>% summarise(total = sum(value))
ggplot(room_usage, aes(x = factor(r), y = total)) +
  geom_bar(stat = "identity") +
  labs(title = "Total Diagnostic Hours by Room", x = "Room", y = "Total Hours")
