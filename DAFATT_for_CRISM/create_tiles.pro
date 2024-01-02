;+
; :Author: Honglei Lin (linhonglei9@gmail.com)



Pro Create_Tiles

  compile_opt idl2
  ENVI,/restore_base_save_files

  envi_batch_init,LOG_FILE='batch.log'

  fname = envi_pickfile(title='Select DAFATT RESULT file(s):',$
    filter='*DAFATT.img',/multiple_files)

  ;main
  for i=0,n_elements(fname)-1 do begin

    print,'Processing',i+1,'file'
    
    envi_open_file,fname[i], R_FID=fid,/INVISIBLE
    if fid EQ -1 then begin
      print,fname[i]+'fail to open file'
    endif

    envi_file_query,fid,DIMS=dims,NB=nb, NL=nl, NS=ns,BNAMES=bnames,fname=datafile, DATA_TYPE= DATA_TYPE
    
    in_file_directory = file_dirname(datafile, /mark_directory)
    in_file_basename = file_basename(datafile)
    poss = strpos(in_file_basename,'.')
    if (poss gt 0) then begin
      in_file_extension = strmid(in_file_basename, poss)
      in_file_basename = strmid(in_file_basename,0,poss)
    endif

    map_info=envi_get_map_info(fid=fid)
    
    values=1
    in_memory = 0
    ;raster to vector 
    ClassImage=fltarr(ns,nl,nb)
        for j=0,nb-1 do begin
          Sum=total(envi_get_data(fid=fid, dims=dims, pos=j)) 
          if Sum eq 0 then begin
              ClassImage[*,*,j]=0
          endif else begin
             
           out_names=FILE_DIRNAME(fname[i])+bnames[j]+'img2vec.evf'
           ;convert detections to Vectors
           ENVI_DOIT, 'rtv_doit', $
            fid=fid, pos=j, dims=dims, $
            IN_MEMORY=in_memory,values=values,out_names=out_names,l_name=bnames[j]
            evf_id=ENVI_EVF_OPEN(out_names)
            output_shapefile_rootname=FILE_DIRNAME(fname[i])+bnames[j]+'vec2shp.shp'
            ENVI_EVF_TO_SHAPEFILE, evf_id, output_shapefile_rootname
            
              e = ENVI()   
              Vector = e.OpenVector(output_shapefile_rootname)    
              Task = ENVITask('VectorRecordsToSeparateROI')
              
              ;Task.ATTRIBUTE_NAME = 'CLASS_NAME'
              Task.INPUT_VECTOR = Vector
              Task.Execute
              Raster = e.OpenRaster(fname[i])
              ;View = e.GetView()
              ;Layer = View.CreateLayer(Raster)
              
             ; Open and display the ROIs
              File = Task.OUTPUT_ROI_URI
              rois = e.OpenRoi(File)
                        
              ;roiLayers = !NULL
              ;FOREACH roi, rois DO $
              ;roiLayers = [roiLayers, Layer.AddRoi(roi)] 
              
               
              ;ROI to classification       
              Task = ENVITask('ROIToClassification')
              Task.INPUT_ROI = Rois ;[Rois[0], Rois[1],Rois[2]]
              Task.INPUT_RASTER = Raster
              Task.Execute  
              e.Data.Add, Task.OUTPUT_RASTER
              ;View = e.GetView()
              ;Layer = View.CreateLayer(Task.OUTPUT_RASTER)
              
              ENVI_OPEN_FILE, Task.OUTPUT_RASTER_URI, R_FID=imagefid
              envi_file_query,imagefid,ns=ns1,nl=nl1,nb=nb1
              image=envi_get_data(fid=imagefid, dims=dims, pos=0) ;
               
               ClassImage[*,*,j]=image
               
               
              DataColl = e.Data
              
              ;DataItems = DataColl.Get()
              ;FOREACH Item, DataItems DO PRINT, Item
              DataColl.Remove,Task.OUTPUT_RASTER
              DataColl.Remove,Vector

              ENVI_EVF_CLOSE, evf_id
              FILE_DELETE, out_names, /quiet 
              FILE_DELETE,Task.OUTPUT_RASTER_URI, /quiet 
              FILE_DELETE, File, /quiet 
              FILE_DELETE, output_shapefile_rootname, /quiet 
              ENVI_FILE_MNG, ID=imagefid, /REMOVE


;              roiLayers = OBJARR(N_ELEMENTS(rois))
;              FOR i=0, N_ELEMENTS(rois)-1 DO roiLayers[i] = layer.AddROI(rois[i]) 
;              roiLayers[1].Close
         endelse 
      endfor
    
  outfile=in_file_directory + in_file_basename + '_tiles' + '.img'
  envi_write_envi_file, ClassImage, BNAMES=bnames,MAP_INFO=map_info, NB=nb, NL=nl, NS=ns, OUT_NAME=outfile
 
  endfor

  envi_batch_exit
end