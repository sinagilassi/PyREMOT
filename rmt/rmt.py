# REACTOR MODELING TOOLS
# -----------------------

# import package/module
import data
import timeit
import docs


# tic
tic = timeit.timeit()

# display feed
print(f"feed mole fraction: {data.feedMoFri}")
print(data.bed_por)

# docs
x = docs.MW_mix
print(f"mixture MW: {x}")

# tac
tac = timeit.timeit()

# computation time [s]
comTime = (tac - tic)*1000
print(f"computation time: {comTime}")
