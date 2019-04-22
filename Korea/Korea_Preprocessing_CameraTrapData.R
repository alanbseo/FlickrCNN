setwd("~/Dropbox/KIT/FlickrEU/FlickrCNN/")


library(stringi)

# stri_trans_general("한글",id = "Any-Latin")
library(stringr)


data_files = list.files("../../../CameraTraps_Korea/Data/", pattern = "xls", full.names = T)
library(readxl)


data_names = c("HwanghakMt.", "Duksanjae", "Gosung", "Geumsan", "Gimchun", "Gunwi", "Moongyeong", "Yeowonjae1", "Yeowonjae2", "Hwangsan1", "Hwangsan2", "Soonchang1", "Soonchang2") # Sachijae with no excel file

site_names = c( "Gosung", "Duksanjae", "Gunwi", "Geumsan", "Moongyeong", "Yeowonjae", "Soonchang1", "Soonchang2", "Gimchun", "Hwangsan", "Sachijae", "HwanghakMt.")



# 황학산 (파일이름없음) test 용..
dt_hwanghak1 = read_excel(data_files[1], sheet = c(1)) # does not have file names  
dt_hwanghak2 = read_excel(data_files[1], sheet = c(2)) # does not have file names   
dt_hwanghak =rbind(dt_hwanghak1, dt_hwanghak2)
table(table(dt_hwanghak$Time)>2) # overlaps.. 


colnames(dt_hwanghak)[4] = "Species_Kor" 
dt_hwanghak$Filename = NA # no file names.. 

# 덕산재
# dt_tmp = rbind(read_excel(data_files[2], sheet = c(1)), read_excel(data_files[2], sheet= 2))# 덕산재는 sheet1이맞은
 # table((table(dt_tmp$파일이름)) > 1)

dt_duksanjae = read_excel(data_files[2], sheet = c(1))
 
colnames(dt_duksanjae)[2] = "Species_Kor"
colnames(dt_duksanjae)[1] = "Filename"
colnames(dt_duksanjae)[7] = "Activity"
colnames(dt_duksanjae)[4] = "Count"
colnames(dt_duksanjae)[5] = "Gender"
colnames(dt_duksanjae)[6] = "Age"
colnames(dt_duksanjae)[11] = "Time"

# 고성
dt_gosung = read_excel(data_files[3], sheet = c(1))
colnames(dt_gosung)[8] = "Species_Kor"
colnames(dt_gosung)[7] = "Filename"
# table(dt_gosung$...6)
nrow(dt_gosung) - length(unique(dt_gosung$Filename))


# 금산
dt_geumsan = read_excel(data_files[4], sheet = c(2)) # Sheet 2 요청통계 
dt_geumsan = dt_geumsan[(dt_geumsan$조사지역 == "금산군"),]
colnames(dt_geumsan)[4] = "Species_Kor"
colnames(dt_geumsan)[3] = "Filename"
colnames(dt_geumsan)[9] = "Activity"
colnames(dt_geumsan)[5] = "Count"
colnames(dt_geumsan)[7] = "Gender"
colnames(dt_geumsan)[8] = "Age"

table(dt_geumsan$조사지역)
table(dt_geumsan$Species_Kor)
# dt_geumsan[dt_geumsan$Species_Kor == "반달가슴곰",]

# 김천 우두령 (시간만있고파일이름없음 파일이이미정리는돼있음)
dt_gimchun = read_excel(data_files[5], sheet = c(1), col_names = F) # does not have file names but already in the folders.. 
dt_gimchun[1,]
colnames(dt_gimchun) = c("Desc", "Site", "Unknown", "Species_Kor", "Date", "DayNight","Time", "IndexMaybe")
table(dt_gimchun$Site)
table(dt_gimchun$Species_Kor)

dt_gimchun$Filename = NA

# Gunwi 
dt_gunwi = read_excel(data_files[6], sheet = c(1), col_names = T)  
dt_gunwi[1,]
colnames(dt_gunwi)
colnames(dt_gunwi)[7] = "Filename"
colnames(dt_gunwi)[8] = "Species_Kor"
table(dt_gunwi$...6)

# 문경
dt_moongyeong = read_excel(data_files[7], sheet = c(1), col_names = T)  
table(dt_moongyeong$주소 == "경상북도 문경시 가은읍 하괴리 산162-1")
dt_moongyeong = dt_moongyeong[dt_moongyeong$주소 == "경상북도 문경시 가은읍 하괴리 산162-1",]

colnames(dt_moongyeong)[2] = "Species_Kor"
colnames(dt_moongyeong)[1] = "Filename"
colnames(dt_moongyeong)[7] = "Activity"
colnames(dt_moongyeong)[4] = "Count"
colnames(dt_moongyeong)[5] = "Gender"
colnames(dt_moongyeong)[6] = "Age"



# 남원 여원재
dt_yeowonjae1 = read_excel(data_files[8], sheet = c(2), col_names = T)  
dt_yeowonjae1 = dt_yeowonjae1[dt_yeowonjae1$주소 == "전라북도 남원시 이백면 양가리 산55-39",]

colnames(dt_yeowonjae1)[3] = "Filename"
colnames(dt_yeowonjae1)[4] = "Species_Kor"
colnames(dt_yeowonjae1)[5] = "Count"
colnames(dt_yeowonjae1)[7] = "Gender"
colnames(dt_yeowonjae1)[8] = "Age"
colnames(dt_yeowonjae1)[9] = "Activity"

table(dt_yeowonjae1$조사지점)
table(dt_yeowonjae1$Species_Kor)



dt_yeowonjae2 = read_excel(data_files[9], sheet = c(1), col_names = T) 
table(dt_yeowonjae2$...6)
colnames(dt_yeowonjae2)[7] = "Filename"
colnames(dt_yeowonjae2)[8] = "Species_Kor"

# 남원 황산
dt_hwangsan1 = read_excel(data_files[10], sheet = c(1), col_names = T)  
dt_hwangsan2 = read_excel(data_files[11], sheet = c(1), col_names = T)  

dt_hwangsan1[1,]
table(dt_hwangsan1$...6)
colnames(dt_hwangsan1)[7] = "Filename"
colnames(dt_hwangsan1)[8] = "Species_Kor"

dt_hwangsan2[1,]
table(dt_hwangsan2$...6)

colnames(dt_hwangsan2)[7] = "Filename"
colnames(dt_hwangsan2)[8] = "Species_Kor"
# dt_hwangsan = rbind(dt_hwangsan1, dt_hwangsan2)
# length(unique(dt_hwangsan$Filename))
# (dt_hwangsan$Filename)

# Soonchang 
dt_soonchang1 = read_excel(data_files[12], sheet = c(2), col_names = T)  
dt_soonchang1 = dt_soonchang1[dt_soonchang1$주소 == "전라북도 순창군 팔덕면 구룡리 산51-1",]

table(dt_soonchang1$조사지점)


colnames(dt_soonchang1)[3] = "Filename"
colnames(dt_soonchang1)[4] = "Species_Kor"
colnames(dt_soonchang1)[5] = "Count"
colnames(dt_soonchang1)[7] = "Gender"
colnames(dt_soonchang1)[8] = "Age"
colnames(dt_soonchang1)[9] = "Activity"

dt_soonchang2 = read_excel(data_files[13], sheet = c(2), col_names = T)  
dt_soonchang2 = dt_soonchang2[dt_soonchang2$주소 == "전라북도 순창군 팔덕면 월곡리 40-6",]
table(dt_soonchang2$조사지점)

colnames(dt_soonchang2)[3] = "Filename"
colnames(dt_soonchang2)[4] = "Species_Kor"
colnames(dt_soonchang2)[5] = "Count"
colnames(dt_soonchang2)[7] = "Gender"
colnames(dt_soonchang2)[8] = "Age"
colnames(dt_soonchang2)[9] = "Activity"

table(dt_hwanghak$Species_Kor)
table(dt_duksanjae$Species_Kor)
table(dt_gosung$Species_Kor)
table(dt_geumsan$Species_Kor)
table(dt_gimchun$Species_Kor)
table(dt_gunwi$Species_Kor)
table(dt_moongyeong$Species_Kor)  
table(dt_yeowonjae1$Species_Kor)
table(dt_hwangsan1$Species_Kor)
table(dt_hwangsan2$Species_Kor)
table(dt_soonchang1$Species_Kor)
table(dt_soonchang2$Species_Kor)

dt_list = list(dt_hwanghak, 
               dt_duksanjae,
               dt_gosung,
               dt_geumsan, 
               dt_gimchun, 
               dt_gunwi, 
               dt_moongyeong, 
               dt_yeowonjae1, 
               dt_yeowonjae2, 
               dt_hwangsan1, 
               dt_hwangsan2,
               dt_soonchang1,
               dt_soonchang2)


newName = function(names) { 
names(table(names))[!names(table(names)) %in% species_tb$Species_Kor]
}


species_tb= read_xlsx("Korea/species_names_info_2019.xlsx", 1)
# mtext(text = camera_kml$NameE, at = coordinates(camera_kml))

sapply(dt_list, FUN = function (x) newName(x$Species_Kor))


allspecies = (unlist(sapply(dt_list[], FUN = function (x) (x$Species_Kor))))
allspecies_v = names(table(allspecies))
allspecies_v = str_replace(allspecies_v,pattern = ",", replacement = "/")

allspecies_final_v = (as.character(unlist(sapply(allspecies_v, FUN = function(x) str_split(x, c("/"))))))

newSpeciesNames = newName(allspecies_final_v)
newSpeciesNames


allspecies_eng_final_v = species_tb$Species_Eng[(match(allspecies_final_v, species_tb$Species_Kor))]
allspecies_latin_final_v = species_tb$Species[(match(allspecies_final_v, species_tb$Species_Kor))]

sort(unique(allspecies_latin_final_v))

allspecies_eng_sorted_v = sort(unique(allspecies_eng_final_v))
allspecies_eng_sorted_v = c(allspecies_eng_sorted_v[!allspecies_eng_sorted_v %in% c("human", "Error", "unidentified")], "human", "unidentified", "Error")


allspecies_latin_sorted_v =  species_tb$Species[(match(allspecies_eng_sorted_v, species_tb$Species_Eng))]
allspecies_kor_sorted_v =  species_tb$Species_Kor[(match(allspecies_eng_sorted_v, species_tb$Species_Eng))]
allspecies_diet_sorted_v =species_tb$Diet[(match(allspecies_eng_sorted_v, species_tb$Species_Eng))]


allspecies_wo_avian_eng_sorted_v = allspecies_eng_sorted_v[ - which(allspecies_diet_sorted_v=="avian") ]
allspecies_wo_avian_eng_sorted_v = allspecies_eng_sorted_v[ - which(allspecies_diet_sorted_v=="avian") ]


# library(openxlsx)
# write.xlsx(newSpeciesNames, file = "Korea/spe.xlsx")

# "," to "/"
spc_kor_l1 = lapply(dt_list[], FUN = function (x) x1 = str_replace(x$Species_Kor, pattern = ",", replacement = "/"))
# string split 
spc_kor_l2 = lapply(spc_kor_l1, FUN = function(x) str_split(x, c("/")))
# kor to eng 
spc_latin_l = lapply(spc_kor_l2, FUN = function(x) sapply(x, FUN = function(x2) species_tb$Species[(match(x2, species_tb$Species_Kor))]))
spc_eng_l = lapply(spc_kor_l2, FUN = function(x) sapply(x, FUN = function(x2) species_tb$Species_Eng[(match(x2, species_tb$Species_Kor))]))


spc_eng_final_l = lapply(spc_eng_l, FUN = function(x_l)
do.call(rbind, lapply(x_l, FUN = function(x) {
  if (length(x)==1) {
    return( c(x, NA, NA))
    } else if (length(x)==2){
      return(c(x, NA))
      } else {
      return(x)
        }
  })
))

spc_latin_final_l = lapply(spc_latin_l, FUN = function(x_l)
  do.call(rbind, lapply(x_l, FUN = function(x) {
    if (length(x)==1) {
      return( c(x, NA, NA))
    } else if (length(x)==2){
      return(c(x, NA))
    } else {
      return(x)
    }
  })
  ))

tb.dummy = numeric(length(allspecies_eng_sorted_v))
names(tb.dummy) = allspecies_eng_sorted_v

spc_eng_prop_df = sapply(spc_eng_final_l, FUN = function(x) {x2 = table(x)/sum(table(x)); tb.dummy[] = 0; tb.dummy[names(x2)]= x2; return(tb.dummy)})

spc_eng_prop_wobirds_df =  spc_eng_prop_df[ - which(allspecies_diet_sorted_v=="avian") , ]
spc_eng_prop_wobirds_df = sapply(1:ncol(spc_eng_prop_wobirds_df), FUN = function(x) spc_eng_prop_wobirds_df[,x] / sum(spc_eng_prop_wobirds_df[,x]))
colSums(spc_eng_prop_wobirds_df)





tb.latin.dummy = numeric(length(allspecies_latin_sorted_v))
names(tb.latin.dummy) = allspecies_latin_sorted_v

spc_latin_prop_df = sapply(spc_latin_final_l, FUN = function(x) {x2 = table(x)/sum(table(x)); tb.latin.dummy[] = 0; tb.latin.dummy[names(x2)]= x2; return(tb.latin.dummy)})
# library(gplots)
# col_species = rich.colors(length(allspecies_eng_sorted_v))

library(RColorBrewer)
n <- length(allspecies_eng_sorted_v)
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_species = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
# pie(rep(1,n), col=sample(col_species, n))
col_species_wo_avian = col_species[ - which(allspecies_diet_sorted_v=="avian") ]


  
  
colnames(spc_eng_prop_wobirds_df) = data_names

# merge Hwangsan and Yeowonjae data
spc_eng_prop_df_reduced = data.frame(spc_eng_prop_wobirds_df)
spc_eng_prop_df_reduced$Yeowonjae = rowMeans(spc_eng_prop_wobirds_df[, c("Yeowonjae1", "Yeowonjae2")])
spc_eng_prop_df_reduced$Hwangsan = rowMeans(spc_eng_prop_wobirds_df[, c("Hwangsan1", "Hwangsan2")])
spc_eng_prop_df_reduced$Yeowonjae1 = spc_eng_prop_df_reduced$Yeowonjae2 = spc_eng_prop_df_reduced$Hwangsan1 = spc_eng_prop_df_reduced$Hwangsan2 = NULL

spc_eng_prop_df_final = spc_eng_prop_df_reduced[, site_names[-11]]

library(openxlsx)
write.xlsx(spc_eng_prop_df_final, file = "Korea/tmp/species_eng_proportions_df.xlsx", row.names=T)

library(rgdal)
camera_kml = readOGR(dsn = "../../../CameraTraps_Korea/Data/CameraTraps_Korea.kml", layer ="딥러닝")
camera_kml



pdf("Korea/tmp/allspecies_eng_sorted_barplot.pdf", width = 12, height = 8)
par(mfrow=c(1,2), mar=c(7,4,2,4))

barplot(as.matrix(spc_eng_prop_df_final), beside=F, col = col_species_wo_avian, names = site_names[-11], las=2, horiz = F)
plot.new()
par(xpd=NA)
legend("center", legend =allspecies_wo_avian_eng_sorted_v, col =  col_species_wo_avian, pch=15, ncol = 2, cex=1.5, bty="n")

dev.off()


### coordinates 
camera_kml$NameK = camera_kml$Name
camera_kml$NameE = str_remove_all(stri_trans_general(camera_kml$NameK, id = "Hang-Latin"), "-")
camera_kml$Name = site_names

camera_kml$Description = NULL
camera_kml$icon = NULL
camera_kml@data = data.frame(Name= camera_kml@data[, c("Name")])

spplot(camera_kml)

str(camera_kml)

spc_eng_prop_sp = camera_kml[,-11]
 

spc_eng_prop_sp@data = cbind(spc_eng_prop_sp@data, rbind(t(spc_eng_prop_df_final[,1:10]), NA,spc_eng_prop_df_final[,11] ))


spc_eng_prop_sp@data$LON = coordinates(spc_eng_prop_sp)[,1]
spc_eng_prop_sp@data$LAT = coordinates(spc_eng_prop_sp)[,2]

# spplot(spc_eng_prop_sp, "roe.deer")

writeOGR(spc_eng_prop_sp, dsn = "Korea/tmp", layer = "CameraTraps2019", driver = "ESRI Shapefile", verbose = T, overwrite_layer = T)

camera_bbox = bbox(camera_kml)

library(OpenStreetMap)
require (rworldmap)
require(rworldxtra)
# library(RgoogleMaps)
# get world map 
nm = c("osm", "maptoolkit-topo",
       "waze", "mapquest", "mapquest-aerial",
       "bing", "stamen-toner", "stamen-terrain",
       "stamen-watercolor", "osm-german", "osm-wanderreitkarte",
       "mapbox", "esri", "esri-topo",
       "nps", "apple-iphoto", "skobbler",
       "opencyclemap", "osm-transport",
       "osm-public-transport", "osm-bbike", "osm-bbike-german")

map = openmap(upperLeft= c(lat= camera_bbox[2,2] + 0.5,   lon= camera_bbox[1,1] - 0.5 ),
              lowerRight = c(lat= camera_bbox[2,1]-0.5 ,   lon= camera_bbox[1,2]+ 0.5 ),
              type="stamen-terrain", zoom = )
map_ll = openproj(map, CRSobj = proj4string(spc_eng_prop_sp))


pdf("Korea/tmp/allspecies_eng_sorted_map.pdf", width = 12, height = 8)

par(mfrow=c(1,2))
# plot(spc_eng_prop_sp)
# plot(getMap(resolution = "high" ), add=T)
plot(map_ll)

mapPies(spc_eng_prop_sp@data, nameX="LON", nameY="LAT", nameZs = colnames(spc_eng_prop_sp@data)[2:21],xlim=c(127,129), ylim=c(35.39, 38.3), add=T, addCatLegend = F, symbolSize = 1.5, zColours = col_species_wo_avian)


text(spc_eng_prop_sp@coords, site_names[], add=T, cex = 0.8, adj = c(0.5,2))
plot.new()
par(xpd=NA)

legend("center", legend =allspecies_wo_avian_eng_sorted_v, col =  col_species_wo_avian, pch=15, ncol = 2, cex=1.2, bty="n")


dev.off()

# ##getting example data
# 
# dataf <- getMap()@data 
# 
# mapBars( dataf, nameX="LON", nameY="LAT" , nameZs=c('GDP_MD_EST',
#                                                     
#                                                     'GDP_MD_EST','GDP_MD_EST') , mapRegion='asia' , symbolSize=2  ,
#          
#          barOrient = 'horiz' )
# 
# 
# mapPies( dataf,nameX="LON", nameY="LAT", nameZs=c('GDP_MD_EST','GDP_MD_EST',
#                                                   
#                                                   'GDP_MD_EST','GDP_MD_EST'),mapRegion='asia', oceanCol = "lightseagreen",
#          
#          landCol = "gray50")
# 


### matching photo files with the species data 

data_files
path_photos = "~/Dropbox/CameraTraps_Korea/"
# data_paths = list.dirs(path_photos, recursive = F, full.names = F)
# data_paths = data_paths[data_paths!="Data"]
data_paths_all = c("HwanghakMt._2013.9.15-2014.1.30", 
               "Jeonbook_MoojooDuksanjae_2017.6.27-2018.4.25",
  "Gangwon_GosungJinbooryeong_2016.10.14-2017.12.6",
  "Chungnam_Geumsan_2016.7.25-2017.12.1",
  "Gyeongbook_GimchunWoodooryeong_2015.9.1-2015.11.12", 
  "Gyeongbook_Gunwi_2017.4.8-2017.12.7", 
  "Gyeongbook_Moongyeong_2017.9.21-2018.4.7", 
  "Jeonbook_NamwonYeowonjae", 
 "Jeonbook_NamwonHwangsan_2017.10.25-2017.11.22", 
 "Jeonbook_NamwonHwangsan_2017.11.22-2017.12.19",
 "Jeonbook_Soonchang_1_2016.8.4-2016.11.30", 
 "Jeonbook_Soonchang_2_2016.8.4-2016.11.30", 
   "Jeonbook_NamwonSachijae")

data_path_names = c("HwanghakMt.", "Duksanjae", "Gosung", "Geumsan", "Gimchun", "Gunwi", "Moongyeong", "Yeowonjae", "Hwangsan1", "Hwangsan2", "Soonchang1", "Soonchang2", "Sachijae") # sachijae with no excel file

dataset_tb = data.frame(Site = data_path_names, File = data_files, Path = data_paths_all)


data_test_idx  = which(dataset_tb$Site%in%c("HwanghakMt.", "Geumsan", "Duksanjae", "Gimchun")) # Gimchun and Hwanghak
data_train_idx = setdiff(1:length(data_names), data_test_idx)

dataset_train_tb =  dataset_tb[ data_train_idx, ]


 


# Good 

data_idx = 3 # Gosung 
table(sapply(dt_list[[data_idx]]$Filename, FUN = function(x) file.exists(paste0(path_photos, dataset_tb[data_idx, "Path"],"/101RECNX/", x))))


data_idx = 6 #Gunwi 
table(sapply(dt_list[[data_idx]]$Filename, FUN = function(x) file.exists(paste0(path_photos, dataset_tb[data_idx, "Path"],"/", x))))

data_idx = 9 # Hwangsan1 
table(sapply(dt_list[[data_idx]]$Filename, FUN = function(x) file.exists(paste0(path_photos, dataset_tb[data_idx, "Path"],"/", x))))
data_idx = 10 # Hwangsan2 
table(sapply(dt_list[[data_idx]]$Filename, FUN = function(x) file.exists(paste0(path_photos, dataset_tb[data_idx, "Path"],"/", x))))

data_idx = 11# Soonchang1
table(sapply(dt_list[[data_idx]]$Filename, FUN = function(x) file.exists(paste0(path_photos, dataset_tb[data_idx, "Path"],"/", x))))


# Bad 
data_idx = 7 # Moongyeong
table(sapply(dt_list[[data_idx]]$Filename, FUN = function(x) file.exists(paste0(path_photos, dataset_tb[data_idx, "Path"],"/", x))))


data_idx = 8 # Jeonbook_NamwonYeowonjae
table(sapply(dt_list[[data_idx]]$Filename, FUN = function(x) file.exists(paste0(path_photos, dataset_tb[data_idx, "Path"],"/", x))))


data_idx = 12# Soonchang2 
table(sapply(dt_list[[data_idx]]$Filename, FUN = function(x) file.exists(paste0(path_photos, dataset_tb[data_idx, "Path"],"/", x))))

data_idx = 13# Sachijae
table(sapply(dt_list[[data_idx]]$Filename, FUN = function(x) file.exists(paste0(path_photos, dataset_tb[data_idx, "Path"],"/", x))))








 
 