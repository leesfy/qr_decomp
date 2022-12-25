PROGRAM main
    implicit none
 
    double precision, dimension (:, :), allocatable :: A
    double precision, dimension (:), allocatable :: tau, work, exmp_work
    integer :: n, i, j, info, lda, lwork
    real ::  start, end
    character(len=32) :: arg


    n = 1000
    CALL get_command_argument(1, arg)
    IF (LEN_TRIM(arg) /= 0) read(arg, *) n

    allocate (A(n, n), tau(n), exmp_work(n))

    do j = 1, n
        do i = 1, n
            A(i, j) = i + j
        end do
    end do


    lda = n
    lwork = -1

    
    call dgeqrf(n, n, A, lda, tau, exmp_work, lwork, info)
    lwork = exmp_work(1)

    allocate(work(lwork))
    call cpu_time(start)    

    call dgeqrf(n, n, A, lda, tau, work, lwork, info)

    call cpu_time(end)
    write (*,*) "Time in seconds: "
    print *, end - start

    deallocate (A, tau, work, exmp_work)!, R, Q, C)

END PROGRAM main