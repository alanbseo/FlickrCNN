
path.wd = "~/Dropbox/KIT/Peatland/FlickrCNN/"

setwd(path.wd)

# Check image validity while re-downloading
checkImage = TRUE

library(readxl)
library(doMC)
library(imager)

n.thread = 5

pause = 1 # wait to avoid network problems..

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


colnames(photos.dt)[c(1,3)] = c("HostID", "URL")
photos.dt$HostID = as.character(photos.dt$HostID)
photos.dt$URL = as.character(photos.dt$URL)

table(photos.dt$Activity)

# photos_50.dt$URL

imgdir <- paste0(path.wd, "/", photos.dir)
if (!dir.exists(imgdir)) { 
    dir.create(imgdir, recursive = T)
}

# parallelising the download process..
registerDoMC(n.thread) 

# photos.dt2 = photos.dt[photos.withtag.idx, ] 
# unique(photos.dt2$PhotoID) # 

# quite a few images with imperfect file ending 
download.res = foreach (p.idx =  (which(photos.withtag.idx)), .inorder = F, .errorhandling = "stop", .verbose = F) %dopar% { 
    
    host.id <- photos.dt$HostID[p.idx]
    photo.url <- photos.dt$URL[p.idx]
    photo.id = paste0(formatC(p.idx, width = 6, flag = "0"), "_", host.id,   "_",   photos.dt$Activity[p.idx], "_", photos.dt$timestamp[p.idx])
    
    
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


