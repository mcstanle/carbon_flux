!This GOSAT XCO2 observation operator was revised by Feng Deng for personal use
!/20120601/

      MODULE GOSAT_CO2_MOD
  
      IMPLICIT NONE 

      !=================================================================
      ! MODULE VARIABLES
      !=================================================================   
 
      ! Parameters
      INTEGER, PARAMETER           :: MAXLEV = 20
      INTEGER, PARAMETER           :: MAXGOS = 2000

      ! Record to store data from each GOS obs
      TYPE GOS_XCO2_OBS 
         INTEGER                      :: LGOS
         REAL*8                       :: LAT
         REAL*8                       :: LON
         REAL*8                       :: HGT
         REAL*8                       :: TIME
         REAL*8                       :: TIME2
         REAL*8                       :: CO2
         REAL*8                       :: CO2_UNC
	 REAL*8			      :: PSURF
	 REAL*8			      :: GCPSURF
         REAL*8                       :: S_OER_INV
         REAL*8                       :: GCco2(47)
         REAL*8                       :: GCP(47)
         !INTEGER		      :: QF
      ENDTYPE GOS_XCO2_OBS  

      TYPE(GOS_XCO2_OBS)              :: GOSX(MAXGOS)

      ! IDTCO2 isn't defined in tracerid_mod because people just assume 
      ! it is one. Define it here for now as a temporary patch. 
      INTEGER, PARAMETER :: IDTCO2   = 1 
      ! Same thing for TCVV(IDTCO2) 
      REAL*8,  PARAMETER :: TCVV_CO2 = 28.97d0  / 44d0 

      CONTAINS
!------------------------------------------------------------------------------
      subroutine read_GOSAT_data(YYYYMMDD, NGOS)
  
      USE FILE_MOD
      USE GRID_MOD,  ONLY : GET_IJ
      USE TIME_MOD,  ONLY : EXPAND_DATE
      USE TRACER_MOD

      INTEGER, INTENT(IN)  :: YYYYMMDD
      INTEGER, PARAMETER   :: iu_analysis = 88 
      INTEGER, PARAMETER   :: maxobs=2000        
      INTEGER              :: NGOS
      INTEGER              :: numobs, i, ios
      integer              :: oday, oyear, omonth, ohour, omin, osec
      integer              :: k, l
      integer              :: iijj(2)
      logical              :: fileopen, it_exists
      character(len=255)   :: READ_FILENAME
      character(len=250)   :: filename
      !-----------------------------------------------------------
      ! get file name 
      !-----------------------------------------------------------

      ! filename root
      READ_FILENAME = TRIM( 'PROTOTYPE_mine_YYYYMMDD.txt' )

      ! Expand date tokens in filename
      CALL EXPAND_DATE( READ_FILENAME, YYYYMMDD, 9999 )

      ! Construct complete filename
      READ_FILENAME = TRIM( '/users/jk/13/bbyrne/python_codes/' ) //
     &                TRIM( READ_FILENAME )

      WRITE(6,*) '    - READ_GOSAT_CO2_OBS: reading file: ',
     &   READ_FILENAME

!      write(filename,100) 
!     &'./df_data/conc_data/XCO2_
!     &YYYYMMDD.txt'
!      CALL EXPAND_DATE(FILENAME, YYYYMMDD, 9999)
      !print*,'filename = ',filename
      !-----------------------------------------------------------
      ! make sure files exists before opening
      !-----------------------------------------------------------
      INQUIRE( FILE=TRIM( READ_FILENAME ), EXIST=IT_EXISTS )
      if (it_exists) then
          open(unit=iu_analysis,file=trim(READ_FILENAME),status='old',
     &    form='formatted')

          fileopen = .true.
          numobs = 1
          ios = 0
          !------------------------------------------------------
          ! loop through data file and read in all retrievals
          !------------------------------------------------------
!          do i=1,9
!             READ ( iu_analysis , *,iostat=ios ) 
!          enddo
      
          do
             READ ( iu_analysis , *,iostat=ios ) 
     &          gosx(numobs)%lon,  
     &          gosx(numobs)%lat, 
     &          gosx(numobs)%hgt, gosx(numobs)%co2, 
     &          gosx(numobs)%co2_unc,oyear,omonth,oday,ohour,
     &          omin,osec


	     gosx(numobs)%co2_unc=gosx(numobs)%co2_unc*0.+0.7d-6
             gosx(numobs)%S_OER_INV=1.0/gosx(numobs)%co2_unc
     &                                    /gosx(numobs)%co2_unc
           
             GOSX(numobs)%TIME = (float(ohour)+(float(omin)+
     &                           (float(osec)/60.0))/60.0)/24.0

             GOSX(numobs)%TIME2 = float(ohour)*10000+float(omin)*100+
     &                            float(osec)

             print*,'SSSSSSSSS'

             print*,'gosx(numobs)%lon',gosx(numobs)%lon
             print*,'gosx(numobs)%lat',gosx(numobs)%lat
             print*,'gosx(numobs)%hgt',gosx(numobs)%hgt
             print*,'gosx(numobs)%co2',gosx(numobs)%co2
             print*,'oyear',oyear
             print*,'omonth',omonth
             print*,'oday',oday
             print*,'ohour',ohour
             print*,'omin',omin
             print*,'osec',osec
             print*,'gosx(numobs)%co2_unc',gosx(numobs)%co2_unc
             print*,'gosx(numobs)%TIME',gosx(numobs)%TIME
             print*,'gosx(numobs)%TIME2',gosx(numobs)%TIME2



             if (ios .gt. 0) then
                stop 'problems reading obs file'
             endif
             if (ios .lt. 0) then
                 print*,'End of GOSAT data file'
                 close(unit=iu_analysis)
                 fileopen = .false.
                 exit
             endif
 
!             read(iu_analysis,*) 
!     &          (gosx(numobs)%co2(i),i=gosx(numobs)%lgos, 1, -1)
!             read(iu_analysis,*) 
!     &          (gosx(numobs)%avg_kern(i),i=gosx(numobs)%lgos, 1, -1)
!             read(iu_analysis,*) 
!     &          (gosx(numobs)%pweight(i),i=gosx(numobs)%lgos, 1, -1)
!             read(iu_analysis,*) 
!     &          (gosx(numobs)%xa(i),i=gosx(numobs)%lgos, 1, -1)
!             read(iu_analysis,*) 
!     &          (gosx(numobs)%pres(i),i=gosx(numobs)%lgos, 1, -1)
!             read(iu_analysis,*)      
!     &          (gosx(numobs)%GCco2(i),i=47, 1, -1)
!             read(iu_analysis,*) 
!     &          (gosx(numobs)%GCP(i),i=47, 1, -1)     

	     gosx(numobs)%co2=gosx(numobs)%co2*1.0d-6
!	     gosx(numobs)%xa(:)=gosx(numobs)%xa(:)*1.0d-6
!	     gosx(numobs)%GCco2(:)=gosx(numobs)%GCco2(:)*1.0d-6
             
!             print*,'!!!!!! STUFF READ IN !!!!!!'
!             print*,'gosx(numobs)%lgos',gosx(numobs)%lgos
!             print*,'gosx(numobs)%lon',gosx(numobs)%lon
!             print*,'gosx(numobs)%lat',gosx(numobs)%lat
!             print*,'gosx(numobs)%psurf',gosx(numobs)%psurf
!             print*,'gosx(numobs)%xco2',gosx(numobs)%xco2
!             print*,'gosx(numobs)%xco2_unc',gosx(numobs)%xco2_unc
!             print*,'oyear',oyear
!             print*,'omonth',omonth
!             print*,'oday',oday
!             print*,'ohour',ohour
!             print*,'gosx(numobs)%GCPSURF',gosx(numobs)%GCPSURF
!             print*,'gosx(numobs)%avg_kern',gosx(numobs)%avg_kern
!             print*,'gosx(numobs)%pweight',gosx(numobs)%pweight
!             print*,'gosx(numobs)%xa',gosx(numobs)%xa
!             print*,'gosx(numobs)%pres',gosx(numobs)%pres
!             print*,'gosx(numobs)%GCco2',gosx(numobs)%GCco2
!             print*,'gosx(numobs)%GCP',gosx(numobs)%GCP
!             print*,'!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        
             WRITE(6,*) 'numobs:',numobs

             numobs = numobs + 1

        enddo     ! end of loop though data file

        NGOS = numobs - 1

        if (fileopen) then
           close(unit=iu_analysis)
	 endif

        else
            print*,'NO XCO2 data files for this date'
            NGOS = 0
        endif

 100  format(a,i4.4,i2.2,i2.2,a)
 110  format(a,i4.4,i2.2,i2.2,a,a)
 120  format(i3,2x,6(1e11.4,1x))

      end subroutine read_GOSAT_data


!------------------------------------------------------------------------------

      SUBROUTINE CALC_GOS_CO2_FORCE( COST_FUNC )
!
!******************************************************************************
!  Subroutine CALC_GOS_CO2_FORCE calculates the adjoint forcing from the GOSAT
!  CO2 observations and updates the cost function. (dkh, 10/12/10) 
! 
!
!  Arguments as Input/Output:
!  ============================================================================
!  (1 ) COST_FUNC (REAL*8) : Cost funciton                        [unitless]
!     
!     
!  NOTES:
!  
!******************************************************************************
!
      ! Reference to f90 modules
      USE ADJ_ARRAYS_MOD,     ONLY : STT_ADJ
      USE ADJ_ARRAYS_MOD,     ONLY : N_CALC
      USE ADJ_ARRAYS_MOD,     ONLY : EXPAND_NAME
      USE CHECKPT_MOD,        ONLY : CHK_STT
      USE COMODE_MOD,         ONLY : CSPEC, JLOP
      USE DAO_MOD,            ONLY : AD
      USE DAO_MOD, 	      ONLY : SPHU
      USE DAO_MOD,            ONLY : AIRDEN
      USE DAO_MOD,            ONLY : BXHEIGHT
      USE DIRECTORY_ADJ_MOD,  ONLY : DIAGADJ_DIR
      USE GRID_MOD,           ONLY : GET_IJ
      USE PRESSURE_MOD,       ONLY : GET_PCENTER, GET_PEDGE
      USE TIME_MOD,           ONLY : GET_NYMD,    GET_NHMS
      USE TIME_MOD,           ONLY : GET_TS_CHEM
      USE TRACER_MOD,         ONLY : XNUMOLAIR
      USE TROPOPAUSE_MOD,     ONLY : ITS_IN_THE_TROP


#     include      "CMN_SIZE"      ! Size params

      ! Arguments
      REAL*8, INTENT(INOUT)       :: COST_FUNC
   
      ! Local variables 
      INTEGER                     :: NTSTART, NTSTOP, NT 
      INTEGER                     :: IIJJ(2), I,      J
      INTEGER                     :: L,       LL,     LGOS
      INTEGER                     :: JLOOP
      REAL*8                      :: GC_CO2_NATIVE
      REAL*8                      :: GC_CO2OBS
      REAL*8                      :: GC_CO2
      REAL*8                      :: CO2_HAT
      REAL*8                      :: CO2_OBS
      REAL*8                      :: CO2_PERT
      REAL*8                      :: CO2_PERT_OBS
      REAL*8                      :: FORCE
      REAL*8                      :: DIFF
      REAL*8                      :: NEW_COST(IIPAR, JJPAR)
      REAL*8                      :: OLD_COST
      REAL*8, SAVE                :: TIME_FRAC(MAXGOS)
      REAL*8, SAVE                :: TIME_HOUR(MAXGOS)
      INTEGER,SAVE                :: NGOS
      REAL*8			  :: ADJ_F
      REAL*8			  :: ADJ_FORCE
      REAL*8                      :: GC_CO2_NATIVE_ADJ
      REAL*8                      :: CO2_HAT_ADJ
      REAL*8                      :: CO2_PERT_ADJ
      REAL*8                      :: GC_CO2_ADJ
      REAL*8                      :: DIFF_ADJ
      LOGICAL, SAVE               :: FIRST = .TRUE. 
      INTEGER                     :: IOS
      CHARACTER(LEN=255)          :: FILENAME



      !=================================================================
      ! CALC_GOS_CO2_FORCE begins here!
      !=================================================================

      print*, '     - CALC_GOS_CO2_FORCE '
    
      ! Reset 
      NEW_COST (:,:) = 0D0 

      ! Open files for diagnostic output
      IF ( FIRST ) THEN

         FILENAME = 'diff.NN.m'
         CALL EXPAND_NAME( FILENAME, N_CALC )
         FILENAME =  TRIM( DIAGADJ_DIR ) //  TRIM( FILENAME )
         OPEN( 105,      FILE=TRIM( FILENAME    ), STATUS='UNKNOWN',
     &       IOSTAT=IOS, FORM='FORMATTED',    ACCESS='SEQUENTIAL' )

      ENDIF

      ! Save a value of the cost function first
      OLD_COST = COST_FUNC

      ! Check if it is the last hour of a day 
      IF ( GET_NHMS() == 236000 - GET_TS_CHEM() * 100 ) THEN 
 
         ! Read the GOS CO2 file for this day 
         CALL  read_GOSAT_data( GET_NYMD(), NGOS ) 


             WRITE(6,*) 'So NGOS is:',NGOS


         ! TIME is YYYYMMDD.frac-of-day.  Subtract date and save just time fraction

         TIME_FRAC(1:NGOS) = GOSX(1:NGOS)%TIME
         TIME_HOUR(1:NGOS) = GOSX(1:NGOS)%TIME2/10000.

      ENDIF 

      ! Get the range of GOS retrievals for the current hour
      CALL GET_NT_RANGE( NGOS, GET_NHMS(), TIME_HOUR, NTSTART, NTSTOP ) 

      IF ( NTSTART == 0 .and. NTSTOP == 0 ) THEN 
         print*, ' No matching GOS XCO2 obs for this hour'
         RETURN
      ENDIF 

      print*, ' for hour range: ', GET_NHMS(), TIME_FRAC(NTSTART), 
     &       TIME_FRAC(NTSTOP)
      print*, ' found record range: ', NTSTART, NTSTOP 

      DO NT  = NTSTART, NTSTOP, -1

         print*, '     - CALC_GOS_CO2_FORCE: analyzing record ', NT 

         ! quality screening
         !IF ( GOS(NT)%QF == 0 ) THEN 
            !print*, ' BAD QF, skipping record ', NT
            !CYCLE
         !ENDIF

         ! For safety, initialize these
         GC_CO2       = 0d0 
         GC_CO2OBS    = 0d0 
         CO2_HAT_ADJ  = 0d0 
         FORCE        = 0d0 
         ADJ_FORCE    = 0d0


!         ! Copy LGOS to make coding a bit cleaner
!         LGOS = GOSX(NT)%LGOS

         ! Get grid box of current record
 !        IIJJ  = GET_IJ( REAL(GOSX(NT)%LON,4), 
 !    &   REAL(GOSX(NT)%LAT,4))
         I     = GOSX(NT)%LON
         J     = GOSX(NT)%LAT
         L     = GOSX(NT)%HGT

         print*, 'I,J,L:', I, J, L


      !   ! Get GC pressure levels (mbar) 
      !   DO L = 1, LLPAR
      !      GC_PRES(L) = GET_PCENTER(I,J,L)
      !   ENDDO
         ! Get GC surface pressure (mbar) 
      !   GC_PSURF = GET_PEDGE(I,J,1)         
         
!         ! Calculate the interpolation weight matrix 
!         MAP(1:LLPAR,1:LGOS) 
!     &      = GET_INTMAP( LLPAR, GC_PRES(:), GC_PSURF, 
!     &                    LGOS,  GOSX(NT)%PRES(1:LGOS), 
!     &                    GOSX(NT)%PSURF )

!         ! Calculate the interpolation weight matrix 
!         MAPG(1:LLPAR,1:LGOS) 
!     &      = GET_INTMAP( LLPAR, GOSX(NT)%GCP(:), GOSX(NT)%GCPSURF, 
!     &                    LGOS,  GOSX(NT)%PRES(1:LGOS), 
!     &                    GOSX(NT)%PSURF )
       
         ! Get CO2 values at native model resolution
         GC_CO2_NATIVE = CHK_STT(I,J,L,IDTCO2)
 
         ! Convert from kg/box to v/v
         GC_CO2_NATIVE = GC_CO2_NATIVE * TCVV_CO2
     &                    / AD(I,J,L) 

         ! Interpolate GC OBS CO2 column to GOSAT grid, LL satellite measurement levels=LGOS, L model levels=LLPAR

!         DO LL = 1, LGOS
!            GC_CO2OBS(LL) = 0d0 
!            DO L = 1, LLPAR 
!               GC_CO2OBS(LL) = GC_CO2OBS(LL) 
!     &                    + MAPG(L,LL) * GOSX(NT)%GCco2(L) 
!            ENDDO
!         ENDDO

         ! Interpolate GC CO2 column to GOSAT grid, LL satellite measurement levels=LGOS, L model levels=LLPAR

!         DO LL = 1, LGOS
!            GC_CO2(LL) = 0d0 
!            DO L = 1, LLPAR 
!               GC_CO2(LL) = GC_CO2(LL) 
!     &                    + MAP(L,LL) * GC_CO2_NATIVE(L) 
!            ENDDO
!         ENDDO


         !calculate modeled column
         
          !--------------------------------------------------------------
         ! Apply GOS observation operator
         !
         !   x_hat = x_a + A_k ( x_m - x_a ) 
         !  
         !  where  
         !    x_hat = GC modeled column as seen by GOSAT [vmr]
         !    x_a   = GOS apriori column                 [vmr]
         !    x_m   = GC modeled column                  [vmr]
         !    A_k   = GOS averaging kernel 
         !--------------------------------------------------------------
            ! x_m - x_a
!         DO L = 1, LGOS 
!           CO2_PERT(L) = GC_CO2(L) - GOSX(NT)%XA(L)
!         ENDDO
!         DO L = 1, LGOS 
!           CO2_PERT_OBS(L) = GC_CO2OBS(L) - GOSX(NT)%XA(L)
!         ENDDO

         ! x_a + A_k * ( x_m - x_a )  

         CO2_HAT=0d0
         CO2_OBS=0d0

  !       print*, 'GC_CO2: ',GC_CO2(1)
  !       print*, 'GOSX(NT)%XA: ',GOSX(NT)%XA(1)
  !       print*, 'GOSX(NT)%PWEIGHT: ',GOSX(NT)%PWEIGHT(1)
  !       print*, 'GOSX(NT)%AVG_KERN:',GOSX(NT)%AVG_KERN(1)
  !       print*, 'CO2_PERT',CO2_PERT(1)
          
  !       print*, 'pwf*xa',GOSX(NT)%PWEIGHT(1)*GOSX(NT)%XA(1)
  !      print*,'z:',GOSX(NT)%PWEIGHT(1)*GOSX(NT)%AVG_KERN(1)*CO2_PERT(1)

 !        DO LL = 1, LGOS
 !           XCO2_OBS = XCO2_OBS+GOSX(NT)%PWEIGHT(LL)*GOSX(NT)%XA(LL) 
 !    &      +GOSX(NT)%PWEIGHT(LL)*GOSX(NT)%AVG_KERN(LL)*CO2_PERT_OBS(LL)
 !        ENDDO

 !        DO LL = 1, LGOS
 !           CO2_HAT = CO2_HAT+GOSX(NT)%PWEIGHT(LL)*GOSX(NT)%XA(LL) 
 !    &      +GOSX(NT)%PWEIGHT(LL)*GOSX(NT)%AVG_KERN(LL)*CO2_PERT(LL)
 !        ENDDO
         
 !        print*, 'cccccccccccccccccc'
 !        print*, 'CO2_HAT:', CO2_HAT
 !        print*, 'XCO2_OBS:', XCO2_OBS

         !--------------------------------------------------------------
         ! Calculate cost function, given S is error in vmr
         ! J = 1/2 [ model - obs ]^T S_{obs}^{-1} [ model - obs ]
         !--------------------------------------------------------------

         ! Calculate difference between modeled and observed XCO2

         DIFF = GC_CO2_NATIVE - GOSX(NT)%CO2!GOSX(NT)%XCO2
       !  DIFF = CO2_HAT - XCO2_OBS!GOSX(NT)%XCO2

         print*, 'YYYYYYYYYYYYYYYYYY' 
         print*, 'GC_CO2_NATIVE:', GC_CO2_NATIVE
         print*, 'GET_NHMS() ', GET_NHMS()
         print*, '----------'
         print*, 'GOSX(NT)%CO2:', GOSX(NT)%CO2
         print*, 'gosx(NT)%TIME2 ', gosx(NT)%TIME2
         print*, '----------'
         print*, 'DIFF:', DIFF

         
                  !Personal requirement: if their difference is greater than 8 ppm, 
         !observation will not be used to constrain surface fluxes 
	   if(abs(DIFF) > 8d-6) then 
	      DIFF=0d0
	      goto 2000
	   endif

          
         ! Calculate 1/2 * DIFF^T * S_{obs}^{-1} * DIFF 
      
          print*, 'GOSX(NT)%S_OER_INV: ',GOSX(NT)%S_OER_INV
          FORCE  =  GOSX(NT)%S_OER_INV * DIFF

          print*, 'FORCE:', FORCE
          print*, 'DIFF AFTER COMP:', DIFF

          NEW_COST(I, J) = NEW_COST(I,J) + 0.5d0 * DIFF* FORCE


         !--------------------------------------------------------------
         ! Begin adjoint calculations 
         !--------------------------------------------------------------

       !  DO L = 1, LGOS 
            DIFF_ADJ=FORCE     
       !  ENDDO


	! adjoint of interpolation 
  !       DO L  = 1, LLPAR
            ADJ_F = 0d0 
  !          DO LL = 1, LGOS
               ADJ_F = DIFF_ADJ
  !          ENDDO
  !       ENDDO

         ! Adjoint of unit conversion 

       	  ADJ_FORCE=ADJ_F*TCVV_CO2/AD(I,J,L)

	
        ! Pass adjoint back to adjoint tracer array
!        DO L=1, LLPAR
            STT_ADJ(I,J,L,IDTCO2) = STT_ADJ(I,J,L,IDTCO2)
     &	    + ADJ_FORCE
!	  ENDDO

      ! Write your diag outputs here
	  WRITE(105,113)  1d6*DIFF

 113    FORMAT(F16.8)

2000  ENDDO  ! NT

      ! Update cost function 
      COST_FUNC = COST_FUNC + SUM(NEW_COST(:,:))

      IF ( FIRST ) FIRST = .FALSE. 

      print*, ' Updated value of COST_FUNC = ', COST_FUNC 
      print*, ' GOS contribution           = ', COST_FUNC - OLD_COST  

      ! Return to calling program
      END SUBROUTINE CALC_GOS_CO2_FORCE



      SUBROUTINE GET_NT_RANGE( NTES, HHMMSS, TIME_HOUR, NTSTART, NTSTOP)
!
!******************************************************************************
!  Subroutine GET_NT_RANGE retuns the range of retrieval records for the 
!  current model hour 
! 
!
!  Arguments as Input:
!  ============================================================================
!  (1 ) NTES   (INTEGER) : Number of TES retrievals in this day 
!  (2 ) HHMMSS (INTEGER) : Current model time 
!  (3 ) TIME_HOUR (REAL) : Vector of times (hour) for the TES retrievals
!     
!  Arguments as Output:
!  ============================================================================
!  (1 ) NTSTART (INTEGER) : TES record number at which to start
!  (1 ) NTSTOP  (INTEGER) : TES record number at which to stop
!     
!  NOTES:
!
!******************************************************************************
!
      ! Reference to f90 modules
      USE ERROR_MOD,    ONLY : ERROR_STOP
      USE TIME_MOD,     ONLY : YMD_EXTRACT

      ! Arguments
      INTEGER, INTENT(IN)   :: NTES
      INTEGER, INTENT(IN)   :: HHMMSS
      REAL*8,  INTENT(IN)   :: TIME_HOUR(NTES)
      INTEGER, INTENT(OUT)  :: NTSTART
      INTEGER, INTENT(OUT)  :: NTSTOP
    
      ! Local variables 
      INTEGER, SAVE         :: NTSAVE
      LOGICAL               :: FOUND_ALL_RECORDS 
      INTEGER               :: NTEST
      INTEGER               :: HH, MM, SS
      REAL*8                :: GC_HH
      REAL*8                :: H1

      !=================================================================
      ! GET_NT_RANGE begins here!
      !=================================================================


      ! Initialize 
      FOUND_ALL_RECORDS  = .FALSE. 
      NTSTART            = 0
      NTSTOP             = 0

      ! set NTSAVE to NTES every time we start with a new file
      IF ( HHMMSS == 230000 ) NTSAVE = NTES
   
      !print*, ' co2 hack : skip lat 100 records, where out of order' 
      !print*, ' co2 hack : skip lat 100 records, where out of order' 
      !print*, ' co2 hack : skip lat 100 records, where out of order' 
      !print*, ' co2 hack : skip lat 100 records, where out of order' 
      !print*, ' co2 hack : skip lat 100 records, where out of order' 
  


      print*, ' GET_NT_RANGE for ', HHMMSS
      print*, ' NTSAVE ', NTSAVE
      print*, ' NTES   ', NTES
   
      CALL YMD_EXTRACT( HHMMSS, HH, MM, SS )


      ! Convert HH from hour to fraction of day 
      GC_HH = REAL(HH,8) 
 
      ! one hour as a fraction of day 
      H1    = 1d0 


      ! dkh debug
      print*, ' co2 time frac = ', TIME_HOUR


    
      ! All records have been read already 
      IF ( NTSAVE == 0 ) THEN 

         print*, 'All records have been read already '
         RETURN 

      ! No records reached yet
      ELSEIF ( TIME_HOUR(NTSAVE) + H1 <= GC_HH ) THEN 
           
      
         print*, 'No records reached yet'
         RETURN

      !
      ELSEIF ( TIME_HOUR(NTSAVE) + H1 >  GC_HH ) THEN 
      
         ! Starting record found
         NTSTART = NTSAVE   

         print*, ' Starting : TIME_HOUR(NTSTART) ', 
     &               TIME_HOUR(NTSTART), NTSTART
 
         ! Now search forward to find stopping record
         NTEST = NTSTART

         DO WHILE ( FOUND_ALL_RECORDS == .FALSE. ) 
              
            ! Advance to the next record
            NTEST = NTEST - 1  
           
            ! Stop if we reach the earliest available record 
            IF ( NTEST == 0 ) THEN 
           
               NTSTOP            = NTEST + 1
               FOUND_ALL_RECORDS = .TRUE.

               print*, ' Records found '
               print*, ' NTSTART, NTSTOP = ', NTSTART, NTSTOP

               ! Reset NTSAVE 
               NTSAVE = NTEST

            ! When the combined test date rounded up to the nearest
            ! half hour is smaller than the current model date, the 
            ! stopping record has been passed. 
            ELSEIF (  TIME_HOUR(NTEST) + H1 <=  GC_HH ) THEN
          
               print*, ' Testing : TIME_HOUR ', 
     &                  TIME_HOUR(NTEST), NTEST
 
               NTSTOP            = NTEST + 1 
               FOUND_ALL_RECORDS = .TRUE. 

               print*, ' Records found '
               print*, ' NTSTART, NTSTOP = ', NTSTART, NTSTOP
                  
               ! Reset NTSAVE 
               NTSAVE = NTEST 

            ELSE 
               print*, ' still looking ', NTEST 
                  
            ENDIF 
                 
         ENDDO 
 
      ELSE

         CALL ERROR_STOP('problem', 'GET_NT_RANGE' ) 

      ENDIF 

      ! Return to calling program
      END SUBROUTINE GET_NT_RANGE

!------------------------------------------------------------------------------

      FUNCTION GET_INTMAP( LGC_TOP, GC_PRESC, GC_SURFP,
     &                     LTM_TOP, TM_PRESC, TM_SURFP  )
     *         RESULT      ( HINTERPZ )
!
!******************************************************************************
!  Function GET_INTMAP linearly interpolates column quatities
!   based upon the centered (average) pressue levels. 
!
!  Arguments as Input:
!  ============================================================================
!  (1 ) LGC_TOP (TYPE) : Description                          [unit]
!  (2 ) GC_PRES (TYPE) : Description                          [unit]
!  (3 ) GC_SURFP(TYPE) : Description                          [unit]
!  (4 ) LTM_TOP (TYPE) : Description                          [unit]
!  (5 ) TM_PRES (TYPE) : Description                          [unit]
!  (6 ) TM_SURFP(TYPE) : Description                          [unit]
!     
!  Arguments as Output:
!  ============================================================================
!  (1 ) HINTERPZ (TYPE) : Description                          [unit]
!     
!  NOTES:
!  (1 ) Based on the GET_HINTERPZ_2 routine I wrote for read_sciano2_mod. 
!
!******************************************************************************
!
      ! Reference to f90 modules
      USE ERROR_MOD,     ONLY : ERROR_STOP
      USE PRESSURE_MOD,  ONLY : GET_BP

      ! Arguments
      INTEGER            :: LGC_TOP, LTM_TOP
      REAL*8             :: GC_PRESC(LGC_TOP)
      REAL*8             :: TM_PRESC(LTM_TOP) 
      REAL*8             :: GC_SURFP
      REAL*8             :: TM_SURFP
 
      ! Return value 
      REAL*8             :: HINTERPZ(LGC_TOP, LTM_TOP)

      ! Local variables 
      INTEGER  :: LGC, LTM
      REAL*8   :: DIFF, DELTA_SURFP
      REAL*8   :: LOW, HI

      !=================================================================
      ! GET_HINTERPZ_2 begins here!
      !=================================================================

      HINTERPZ(:,:) = 0D0 
  
!      ! Rescale GC grid according to TM surface pressure
!!         p1_A =     (a1 + b1 (ps_A - PTOP))
!!         p2_A =     (a2 + b2 (ps_A - PTOP))
!!         p1_B =     (a + b (ps_B - PTOP))
!!         p2_B =    *(a + b (ps_B - PTOP))
!!         pc_A = 0.5(a1+a2 +(b1+b2)*(ps_A - PTOP))
!!         pc_B = 0.5(a1+a2 +(b1+b2)*(ps_B - PTOP))
!!         pc_B - pc_A = 0.5(b1_b2)(ps_B-ps_A)
!!         pc_B = 0.5(b1_b2)(ps_B-ps_A) + pc_A
      DELTA_SURFP   = 0.5d0 * ( TM_SURFP -GC_SURFP )

      DO LGC = 1, LGC_TOP
         GC_PRESC(LGC) = ( GET_BP(LGC) + GET_BP(LGC+1))
     &               * DELTA_SURFP + GC_PRESC(LGC)
         IF (GC_PRESC(LGC) < 0) THEN 
            CALL ERROR_STOP( 'highly unlikey', 
     &                       'read_sciano2_mod.f')
         ENDIF 

      ENDDO 
      

      ! Loop over each pressure level of TM grid
      DO LTM = 1, LTM_TOP
 
         ! Find the levels from GC that bracket level LTM
         DO LGC = 1, LGC_TOP - 1

            LOW = GC_PRESC(LGC+1)
            HI  = GC_PRESC(LGC)
            !IF (LGC == 0) HI = TM_SURFP !This is impossible dengf

            ! Linearly interpolate value on the LTM grid 
            IF ( TM_PRESC(LTM) <= HI .and. 
     &           TM_PRESC(LTM)  > LOW) THEN 

               DIFF                = HI - LOW  
               HINTERPZ(LGC+1,LTM) = ( HI - TM_PRESC(LTM)  ) / DIFF
               HINTERPZ(LGC  ,LTM) = ( TM_PRESC(LTM) - LOW ) / DIFF


            ENDIF 
 
            ! dkh debug
            !print*, 'LGC,LTM,HINT', LGC, LTM, HINTERPZ(LGC,LTM)

          ENDDO
       ENDDO

       ! Bug fix:  a more general version allows for multiples TES pressure
       ! levels to exist below the lowest GC pressure.  (dm, dkh, 09/30/10) 
       ! OLD code:
       !IF ( TM_PRESC(1) > GC_PRESC(1) ) THEN
       !   HINTERPZ(1,1)         = 1D0 
       !   HINTERPZ(2:LGC_TOP,1) = 0D0 
       !ENDIF
       ! New code:
       ! Loop over each pressure level of TM grid
       DO LTM = 1, LTM_TOP
          IF ( TM_PRESC(LTM) > GC_PRESC(1) ) THEN
             HINTERPZ(1,LTM)         = 1D0
             HINTERPZ(2:LGC_TOP,LTM) = 0D0
          ENDIF
       ENDDO

      ! Return to calling program
      END FUNCTION GET_INTMAP


      FUNCTION GET_IJ_2x25( LON, LAT ) RESULT ( IIJJ )

!
!******************************************************************************
!  Subroutine GET_IJ_2x25 returns I and J index from the 2 x 2.5 grid for a 
!  LON, LAT coord. (dkh, 11/08/09) 
! 
!
!  Arguments as Input:
!  ============================================================================
!  (1 ) LON (REAL*8) : Longitude                          [degrees]
!  (2 ) LAT (REAL*8) : Latitude                           [degrees]
!     
!  Function result
!  ============================================================================
!  (1 ) IIJJ(1) (INTEGER) : Long index                    [none]
!  (2 ) IIJJ(2) (INTEGER) : Lati index                    [none]
!     
!  NOTES:
!
!******************************************************************************
!     
      ! Reference to f90 modules
      USE ERROR_MOD,    ONLY : ERROR_STOP

      ! Arguments
      REAL*4    :: LAT, LON
      
      ! Return
      INTEGER :: I, J, IIJJ(2)
      
      ! Local variables 
      REAL*8              :: TLON, TLAT, DLON, DLAT
      REAL*8,  PARAMETER  :: DISIZE = 2.5d0
      REAL*8,  PARAMETER  :: DJSIZE = 2.0d0
      INTEGER, PARAMETER  :: IIMAX  = 144
      INTEGER, PARAMETER  :: JJMAX  = 91
      
      
      !=================================================================
      ! GET_IJ_2x25 begins here!
      !=================================================================

      TLON = 180d0 + LON + DISIZE
      TLAT =  90d0 + LAT + DJSIZE
      
      I = TLON / DISIZE
      J = TLAT / DJSIZE

      
      IF ( TLON / DISIZE - REAL(I)  >= 0.5d0 ) THEN
         I = I + 1
      ENDIF
      
      IF ( TLAT / DJSIZE - REAL(J)  >= 0.5d0 ) THEN
         J = J + 1
      ENDIF

      
      ! Longitude wraps around
      !IF ( I == 73 ) I = 1 
      IF ( I == ( IIMAX + 1 ) ) I = 1
      
      ! Check for impossible values 
      IF ( I > IIMAX .or. J > JJMAX .or. 
     &     I < 1     .or. J < 1          ) THEN
         CALL ERROR_STOP('Error finding grid box', 'GET_IJ_2x25')
      ENDIF
      
      IIJJ(1) = I
      IIJJ(2) = J
      
      ! Return to calling program
      END FUNCTION GET_IJ_2x25

!------------------------------------------------------------------------------

  
!------------------------------------------------------------------------------
*
*     Auxiliary routine: printing a matrix.
*
      SUBROUTINE PRINT_MATRIX( DESC, M, N, A, LDA )
      CHARACTER*(*)    DESC
      INTEGER          M, N, LDA
      DOUBLE PRECISION A( LDA, * )
*
      INTEGER          I, J
*
      WRITE(*,*)
      WRITE(*,*) DESC
      DO I = 1, M
         WRITE(*,9998) ( A( I, J ), J = 1, N )
      END DO
*
! Change format of output (dkh, 05/04/10) 
! 9998 FORMAT( 11(:,1X,F6.2) )
 9998 FORMAT( 11(:,1X,E14.8) )
      RETURN

      END SUBROUTINE PRINT_MATRIX 
!------------------------------------------------------------------------------

      END MODULE GOSAT_CO2_MOD
