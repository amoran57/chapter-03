#semiconductor data: predicting chip failure
SC <- read.csv("semiconductor.csv")
full <- glm(FAIL ~ ., data=SC, family = binomial)
#note that we have a "perfect fit" warning; this can indicate overfit
1 - full$deviance/full$null.deviance
#R^2 is 0.56

## grab p-values
pvals <- summary(full)$coef[-1,4] #-1 to drop the intercept
## plot them: it looks like we have some signal here
hist(pvals, xlab="p-value", main="", col="lightblue")

#perform an FDR cut on the p-values
fdr_cut <- function(pvals, q=0.1){
  #create pvalues from input
  pvals <- sort(pvals[!is.na(pvals)])
  N <- length(pvals)
  k <- rank(pvals, ties.method="min")
  #alpha is the maximum pvalue which satisfies the following
  alpha <- max(pvals[ pvals<= (q*k/(N+1)) ])
  
  #plot the pvalues on a log-log graph
  plot(pvals, log="xy", xlab="order", main=sprintf("FDR of %g",q),
       ylab="p-value", bty="n", col=c(8,2)[(pvals<=alpha) + 1], pch=20)
  lines(1:N, q*(1:N)/(N+1))
  
  return(alpha)
}

fdr_cut(pvals)
#our alpha is 0.01217043: we expect that 10% of pvalues below this value are false signals
#but that 90% are true signals

#we identify those 25 pvalues which fall below the cutoff
signif <- which(pvals <= 0.0122)
head (signif)
cutvar <- c("FAIL", names(signif))
cut <- glm(FAIL ~ ., data=SC[,cutvar], family="binomial")
1 - cut$deviance/cut$null.deviance
#so our R^2 is now 0.18--much lower than before, but we don't care about
#in-sample R^2. We want to know how well our model works out of sample.

## Out of sample prediction experiment
## first, define the deviance and R2 functions

## pred must be probabilities (0<pred<1) for binomial
deviance <- function(y, pred, family=c("gaussian","binomial")){
  family <- match.arg(family)
  if(family=="gaussian"){
    return( sum( (y-pred)^2 ) )
  }else{
    if(is.factor(y)) y <- as.numeric(y)>1
    return( -2*sum( y*log(pred) + (1-y)*log(1-pred) ) )
  }
}

## get null deviance too, and return R2
R2 <- function(y, pred, family=c("gaussian","binomial")){
  fam <- match.arg(family)
  if(fam=="binomial"){
    if(is.factor(y)){ y <- as.numeric(y)>1 }
  }
  dev <- deviance(y, pred, family=fam)
  dev0 <- deviance(y, mean(y), family=fam)
  return(1-dev/dev0)
}

# setup the experiment
n <- nrow(SC) # the number of observations
K <- 10 # the number of `folds'
# create a vector of fold memberships (random order)
foldid <- rep(1:K,each=ceiling(n/K))[sample(1:n)]
# create an empty dataframe of results
OOS <- data.frame(full=rep(NA,K), cut=rep(NA,K)) 
# use a for loop to run the experiment
for(k in 1:K){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ## fit the two regressions
  rfull <- glm(FAIL~., data=SC, subset=train, family=binomial)
  rcut <- glm(FAIL~., data=SC[,cutvar], subset=train, family=binomial)
  
  ## get predictions: type=response so we have probabilities
  predfull <- predict(rfull, newdata=SC[-train,], type="response")
  predcut <- predict(rcut, newdata=SC[-train,], type="response")
  
  ## calculate and log R2
  OOS$full[k] <- R2(y=SC$FAIL[-train], pred=predfull, family="binomial")
  OOS$cut[k] <- R2(y=SC$FAIL[-train], pred=predcut, family="binomial")
  
  ## print progress
  cat(k, " ")
}
## plot it in plum
par(mai=c(.9,.9,.1,.1))
boxplot(OOS, col="plum", ylab="R2", xlab="model", bty="n")

#forward step-wise model:
null <- glm(FAIL~1, data=SC)
system.time (fwd <- step(null, scope=formula(full), dir="forward"))
#this was great, and we got 70 potential models. 
#BUT, it took 104 seconds to run, on a relatively small dataset


#working with lasso technique on the browser data

library(gamlr)

## Browsing History. 
## The table has three colums: [machine] id, site [id], [# of] visits
web <- read.csv("browser-domains.csv")
## Read in the actual website names and relabel site factor
sitenames <- scan("browser-sites.txt", what="character")
web$site <- factor(web$site, levels=1:length(sitenames), labels=sitenames)
## also factor machine id
web$id <- factor(web$id, levels=1:length(unique(web$id)))

## get total visits per-machine and % of time on each site
## tapply(a,b,c) does c(a) for every level of factor b.
machinetotals <- as.vector(tapply(web$visits,web$id,sum)) 
visitpercent <- 100*web$visits/machinetotals[web$id]

## use this info in a sparse matrix
## this is something you'll be doing a lot; familiarize yourself.
xweb <- sparseMatrix(
  i=as.numeric(web$id), j=as.numeric(web$site), x=visitpercent,
  dims=c(nlevels(web$id),nlevels(web$site)),
  dimnames=list(id=levels(web$id), site=levels(web$site)))

## now read in the spending data 
yspend <- read.csv("browser-totalspend.csv", row.names=1)  # us 1st column as row names
yspend <- as.matrix(yspend) ## good practice to move from dataframe to matrix

## run a lasso path plot
spender <- gamlr(xweb, log(yspend), verb=TRUE)
plot(spender) ## path plot

#we can also use lasso on logistic regressions
# for gamlr, and most other functions, you need to create your own numeric
# design matrix.  We'll do this as a sparse `simple triplet matrix' using 
# the sparse.model.matrix function.
scx <- sparse.model.matrix(FAIL ~ ., data=SC)[,-1] # do -1 to drop intercept!
# here, we could have also just done x <- as.matrix(SC[,-1]).
# but sparse.model.matrix is a good way of doing things if you have factors.
scy <- SC$FAIL # pull out `y' too just for convenience

# fit a single lasso
sclasso <- gamlr(scx, scy, family="binomial")
plot(sclasso) # the ubiquitous path plot

#import OJ data
oj <- read.csv("oj.csv")
xbrand <- sparse.model.matrix(~brand, data=oj)
xbrand[c(100,200,300),]
#with penalization (eg the lasso path), factor reference levels now matter!
#we can't just absorb one brand into the intercept--we need 3 dummys for 3 brands
xnaref <- function(x){
  if(is.factor(x))
    if(!is.na(levels(x)[1]))
      x <- factor(x,levels=c(NA,levels(x)),exclude=NULL)
    return(x) }

naref <- function(DF){
  if(is.null(dim(DF))) return(xnaref(DF))
  if(!is.data.frame(DF)) 
    stop("You need to give me a data.frame or a factor")
  DF <- lapply(DF, xnaref)
  return(as.data.frame(DF))
}
#this should fix our factor reference problem!
oj$brand <- naref(oj$brand)
xbrand <- sparse.model.matrix(~brand,data=oj)
xbrand[c(100,200,300),]

#one other thing: size MATTERS. We want to scale our parameters equally, 
#since the penalties are size-based