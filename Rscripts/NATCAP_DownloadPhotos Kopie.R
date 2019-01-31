
path.wd = "~/Dropbox/KIT/Peatland/FlickrCNN/"

setwd(path.wd)

# Check image validity while re-downloading
checkImage = TRUE

library(readxl)
library(doMC)
library(imager)

n.thread = 3

pause = 2 # wait to avoid network problems..

# mbsmiddlefork_flickrimages_rand50totagactivities

# Randomly sampled 50 photos
photos.dt = read_xlsx("Data/ActivityTags_EHL.xlsx", sheet = 2)
photos.dt$time = as.POSIXct(photos.dt$timestamp, origin="1970-01-01", tz = "US/Pacific") # timezone? 


photos.withtag.idx = !is.na(photos.dt$Activity)
n_photos_tagged = nrow(photos.dt[photos.withtag.idx,])
photos.dir = paste0("Photos_", n_photos_tagged)


 # All photos
# photos.dt = read.csv("Data/mbsmiddlefork_flickrimages.csv", header = T)
# photos.dir = "Photos_All"


colnames(photos.dt)[c(1,3)] = c("PhotoID", "URL")
photos.dt$PhotoID = as.character(photos.dt$PhotoID)
photos.dt$URL = as.character(photos.dt$URL)

# photos_50.dt$URL

imgdir <- paste0(path.wd, "/", photos.dir)
if (!dir.exists(imgdir)) { 
    dir.create(imgdir, recursive = T)
}

# parallelising the download process..
registerDoMC(n.thread) 


# p.idx =18 throws the 404 error 
download.res = foreach (p.idx = which(photos.withtag.idx), .inorder = F, .errorhandling = "stop", .verbose = F) %dopar% { 
 
    photo.id <- photos.dt$PhotoID[p.idx]
    photo.url <- photos.dt$URL[p.idx]
 
    temp <- paste(imgdir, "/photoid_", photo.id, ".jpg", sep="")
    
    if (!file.exists(temp)) {
        cat("Download_photoid_", photo.id, " (p.idx=", p.idx, ") > ")
        
        # dealing with the 404 error etc. 
        tryCatch(download.file(photo.url, temp, mode="wb", cacheOK = T), error= function(e) {print(e); return(FALSE)})
        
        Sys.sleep(pause)
        
        return(TRUE) 
   
         } else { 
        
         
        if (checkImage) { 
            img <- imager::load.image(temp)
            
            if (!is.cimg(img)) {
                print("re-download the file")
                cat("Download_photoid_", photo.id, " > ")
                
                tryCatch(download.file(photo.url, temp, mode="wb", cacheOK = T), error= function(e) {print(e); return(FALSE)})            
            } else {
                # cat(photo.id, "_s>")
                cat(".")
             }
            
        } else {
            cat(".")
            
        }
        Sys.sleep(pause)
        
        return(TRUE)
    }
}

summary(unlist(download.res))


