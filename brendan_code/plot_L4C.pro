MTYPE = CTM_TYPE( 'GEOS5_47L', RESOLUTION=4 )
GRIDINFO4x5 = CTM_GRID( MTYPE )
XMID4x5 = GRIDINFO4x5.XMID
YMID4x5 = GRIDINFO4x5.YMID

MTYPE = CTM_TYPE( 'GEOS5_47L', RESOLUTION=2 )
GRIDINFO2x25 = CTM_GRID( MTYPE )
XMID2x25 = GRIDINFO2x25.XMID
YMID2x25 = GRIDINFO2x25.YMID
grid_area2x25 = CTM_BOXSIZE( GRIDINFO2x25, /M2 )  

MTYPE = CTM_TYPE( 'GENERIC', RESOLUTION=1 )
GRIDINFO1x1 = CTM_GRID( MTYPE )
XMID1x1 = GRIDINFO1x1.XMID
YMID1x1 = GRIDINFO1x1.YMID
grid_area1x1 = CTM_BOXSIZE( GRIDINFO1x1, /M2 )  


   CTM_GETWEIGHT, GRIDINFO1x1, GRIDINFO4x5, WEIGHT, XX_IND, YY_IND, $
                  WEIGHTFILE = 'weights.1x1.to.geos1.4x5'

   CTM_GETWEIGHT, GRIDINFO1x1, GRIDINFO2x25, WEIGHT, XX_IND, YY_IND, $
                  WEIGHTFILE = 'weights.1x1.to.geos1.2x25'

; gC/m2/day to molec CO2/cm2/sec

;gC/m2/day * 1mol/12g * 6.022×10^23molec/1mol * (m/100cm)^2 * 1day/(24*60*60)s
convert = 6.022E23/(12.*100.*100.*24.*60.*60.)

dir_path = '/users/jk/17/bbyrne/data/for_Zhihua/L4C4CMSnc/NatureRun/Y2002/L4C_NatureRun_NEE_20020601.nc'
dir_path = '/users/jk/17/bbyrne/data/for_Zhihua/L4C4CMSnc/NatureRun/Y2009/'
datafiles = FILE_SEARCH(dir_path,'*.nc')

nfiles = SIZE(datafiles,/N_ELEMENTS)

flux_timeseries = MAKE_ARRAY(nfiles)
flux_timeseries2x25 = MAKE_ARRAY(nfiles)
FOR i = 0, nfiles-1 DO BEGIN

print, STRMID(datafiles[i],84,7)

f2016 = '/home/bbyrne/data/for_Zhihua/L4C4CMSnc/OpsRunV4040/L4C_OpsRun_NEE_2016'+STRMID(datafiles[i],84,7)
fileID = NCDF_Open(f2016, /NOWRITE)
vID = NCDF_VARID(fileid, 'latitude')
NCDF_VARGET, fileId, vId, latitude
vID = NCDF_VARID(fileid, 'longitude')
NCDF_VARGET, fileId, vId, longitude
vID = NCDF_VARID(fileid, 'layer')
NCDF_VARGET, fileId, vId, flux2016
NCDF_CLOSE, fileID
flux2016 = REVERSE(flux2016,2)
flux2016[WHERE(flux2016 LT -1e3)] = 0

f2017 = '/home/bbyrne/data/for_Zhihua/L4C4CMSnc/OpsRunV4040/L4C_OpsRun_NEE_2017'+STRMID(datafiles[i],84,7)
fileID = NCDF_Open(f2017, /NOWRITE)
vID = NCDF_VARID(fileid, 'latitude')
NCDF_VARGET, fileId, vId, latitude
vID = NCDF_VARID(fileid, 'longitude')
NCDF_VARGET, fileId, vId, longitude
vID = NCDF_VARID(fileid, 'layer')
NCDF_VARGET, fileId, vId, flux2017
NCDF_CLOSE, fileID
flux2017 = REVERSE(flux2017,2)
flux2017[WHERE(flux2017 LT -1e3)] = 0

f2018 = '/home/bbyrne/data/for_Zhihua/L4C4CMSnc/OpsRunV4040/L4C_OpsRun_NEE_2018'+STRMID(datafiles[i],84,7)
fileID = NCDF_Open(f2018, /NOWRITE)
vID = NCDF_VARID(fileid, 'latitude')
NCDF_VARGET, fileId, vId, latitude
vID = NCDF_VARID(fileid, 'longitude')
NCDF_VARGET, fileId, vId, longitude
vID = NCDF_VARID(fileid, 'layer')
NCDF_VARGET, fileId, vId, flux2018
NCDF_CLOSE, fileID
flux2018 = REVERSE(flux2018,2)
flux2018[WHERE(flux2018 LT -1e3)] = 0

fluxmean = (flux2016 + flux2017 + flux2018)/3.
fluxmean = fluxmean*convert



   NEP_4x5 = MAKE_ARRAY(72,46, /DOUBLE, VALUE = 0)
   NEP_2x25 = MAKE_ARRAY(144,91, /DOUBLE, VALUE = 0)
   IF (i EQ 0) THEN BEGIN
      NEP_4x5 = CTM_REGRIDH(fluxmean, GRIDINFO1x1, GRIDINFO4x5,/VERBOSE, WFILE='weights.1x1.to.geos1.4x5',/PER_UNIT_AREA)
      NEP_2x25 = CTM_REGRIDH(fluxmean, GRIDINFO1x1, GRIDINFO2x25,/VERBOSE, WFILE='weights.1x1.to.geos1.2x25',/PER_UNIT_AREA)
   ENDIF ELSE BEGIN
      NEP_4x5 = CTM_REGRIDH(fluxmean, GRIDINFO1x1, GRIDINFO4x5,/VERBOSE,/PER_UNIT_AREA)
      NEP_2x25 = CTM_REGRIDH(fluxmean, GRIDINFO1x1, GRIDINFO2x25,/VERBOSE,/PER_UNIT_AREA)
   ENDELSE

   dirout = '/users/jk/17/bbyrne/L4C_NEP/'
   fname = 'nep.geos.2x25.'+string(i+1,FORMAT='(I03)')
   fnameout = dirout + fname
   ; ----
   month_outw = 19850000 + DOUBLE(STRMID(datafiles[i],84,7))
   print, month_outw
   ; ----
   DATAINFO = CREATE3DHSTRU( [8] )
   FILEINFO = CREATE3DFSTRU( [8] )
                                ; ----                                                                                                                                                                                            
   TrcOut = 40005
                                ; ----                                                                                                                                                                                            
   RESULT = CTM_MAKE_DATAINFO( NEP_2x25, DATAINFO, FILEINFO,        $
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=2 ), DIAGN='CO2-SRCE',             $ 
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,000000), TAU1=NYMD2TAU(month_outw,030000), $
                               UNIT='molec CO2/cm2', DIM=[144,91,1,0],                            $
                               TOPTITLE='3-Hour CO2 Flux')
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout
   RESULT = CTM_MAKE_DATAINFO( NEP_2x25, DATAINFO, FILEINFO,        $ 
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=2 ), DIAGN='CO2-SRCE',             $
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,030000), TAU1=NYMD2TAU(month_outw,060000), $
                               UNIT='molec CO2/cm2', DIM=[144,91,1,0],                            $
                               TOPTITLE='3-Hour CO2 Flux')
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND          
   RESULT = CTM_MAKE_DATAINFO( NEP_2x25, DATAINFO, FILEINFO,        $ 
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=2 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,060000), TAU1=NYMD2TAU(month_outw,090000), $
                               UNIT='molec CO2/cm2', DIM=[144,91,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND
   RESULT = CTM_MAKE_DATAINFO( NEP_2x25, DATAINFO, FILEINFO,        $
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=2 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,090000), TAU1=NYMD2TAU(month_outw,120000), $
                               UNIT='molec CO2/cm2', DIM=[144,91,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')  
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND
   RESULT = CTM_MAKE_DATAINFO( NEP_2x25, DATAINFO, FILEINFO,        $                                                                                                                      
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=2 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,120000), TAU1=NYMD2TAU(month_outw,150000), $
                               UNIT='molec CO2/cm2', DIM=[144,91,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')  
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND
   RESULT = CTM_MAKE_DATAINFO( NEP_2x25, DATAINFO, FILEINFO,        $
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=2 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,150000), TAU1=NYMD2TAU(month_outw,170000), $
                               UNIT='molec CO2/cm2', DIM=[144,91,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND
   RESULT = CTM_MAKE_DATAINFO( NEP_2x25, DATAINFO, FILEINFO,        $
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=2 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,180000), TAU1=NYMD2TAU(month_outw,210000), $
                               UNIT='molec CO2/cm2', DIM=[144,91,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')  
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND
   RESULT = CTM_MAKE_DATAINFO( NEP_2x25, DATAINFO, FILEINFO,        $
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=2 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,210000), TAU1=NYMD2TAU(month_outw,240000), $
                               UNIT='molec CO2/cm2', DIM=[144,91,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')           
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND                                                                                                                                   ; ----    

   fname = 'nep.geos.4x5.'+string(i+1,FORMAT='(I03)')
   fnameout = dirout + fname
   ; ----
   month_outw = 19850000 + DOUBLE(STRMID(datafiles[i],84,7))
   print, month_outw
   ; ----
   DATAINFO = CREATE3DHSTRU( [8] )
   FILEINFO = CREATE3DFSTRU( [8] )
                                ; ----                                                                                                                                                                                            
   TrcOut = 40005
                                ; ----                                                                                                                                                                                            
   RESULT = CTM_MAKE_DATAINFO( NEP_4x5, DATAINFO, FILEINFO,        $
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=4 ), DIAGN='CO2-SRCE',             $ 
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,000000), TAU1=NYMD2TAU(month_outw,030000), $
                               UNIT='molec CO2/cm2', DIM=[72,46,1,0],                            $
                               TOPTITLE='3-Hour CO2 Flux')
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout
   RESULT = CTM_MAKE_DATAINFO( NEP_4x5, DATAINFO, FILEINFO,        $ 
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=4 ), DIAGN='CO2-SRCE',             $
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,030000), TAU1=NYMD2TAU(month_outw,060000), $
                               UNIT='molec CO2/cm2', DIM=[72,46,1,0],                            $
                               TOPTITLE='3-Hour CO2 Flux')
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND          
   RESULT = CTM_MAKE_DATAINFO( NEP_4x5, DATAINFO, FILEINFO,        $ 
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=4 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,060000), TAU1=NYMD2TAU(month_outw,090000), $
                               UNIT='molec CO2/cm2', DIM=[72,46,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND
   RESULT = CTM_MAKE_DATAINFO( NEP_4x5, DATAINFO, FILEINFO,        $
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=4 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,090000), TAU1=NYMD2TAU(month_outw,120000), $
                               UNIT='molec CO2/cm2', DIM=[72,46,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')  
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND
   RESULT = CTM_MAKE_DATAINFO( NEP_4x5, DATAINFO, FILEINFO,        $                                                                                                                      
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=4 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,120000), TAU1=NYMD2TAU(month_outw,150000), $
                               UNIT='molec CO2/cm2', DIM=[72,46,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')  
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND
   RESULT = CTM_MAKE_DATAINFO( NEP_4x5, DATAINFO, FILEINFO,        $
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=4 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,150000), TAU1=NYMD2TAU(month_outw,170000), $
                               UNIT='molec CO2/cm2', DIM=[72,46,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND
   RESULT = CTM_MAKE_DATAINFO( NEP_4x5, DATAINFO, FILEINFO,        $
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=4 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,180000), TAU1=NYMD2TAU(month_outw,210000), $
                               UNIT='molec CO2/cm2', DIM=[72,46,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')  
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND
   RESULT = CTM_MAKE_DATAINFO( NEP_4x5, DATAINFO, FILEINFO,        $
                               MODEL=CTM_TYPE( 'GEOS5', RESOLUTION=4 ), DIAGN='CO2-SRCE',             $            
                               TRACER=TrcOut, TAU0=NYMD2TAU(month_outw,210000), TAU1=NYMD2TAU(month_outw,240000), $
                               UNIT='molec CO2/cm2', DIM=[72,46,1,0],                            $                
                               TOPTITLE='3-Hour CO2 Flux')           
   ctm_writebpch,DATAINFO,FILEINFO,filename=fnameout,/APPEND                                                                                                                                   ; ----    

flux_timeseries[i] = TOTAL(fluxmean[*,129:154]*grid_area1x1[*,129:154])
flux_timeseries2x25[i] = TOTAL(NEP_2x25[*,65:77]*grid_area2x25[*,65:77])
ENDFOR

WINDOW, 4
plot, flux_timeseries*(1000./(60.*60.*24.)), yrange=[-4e25, 2e25], ystyle=1
oplot, flux_timeseries2x25*(1000./(60.*60.*24.)), color=2

stophere

;WINDOW, 0
;TvMap, flux, XMID1x1, YMID1x1, /Sample, /Coasts, /CBar, MINDATA=-5, MAXDATA=5;

;WINDOW, 1
;TvMap, NEP_2x25, XMID2x25, YMID2x25, /Sample, /Coasts, /CBar, MINDATA=-5, MAXDATA=5

;WINDOW, 2
;TvMap, NEP_4x5, XMID4x5, YMID4x5, /Sample, /Coasts, /CBar, MINDATA=-5, MAXDATA=5

END
