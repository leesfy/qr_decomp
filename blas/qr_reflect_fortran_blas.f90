PROGRAM main
    implicit none
 
    double precision, dimension (:, :), allocatable :: A, R, Q
    integer :: n, i, j, l
    real ::  start, end
    character(len=32) :: arg


    n = 1000
    CALL get_command_argument(1, arg)
    IF (LEN_TRIM(arg) /= 0) read(arg, *) n

    allocate (A(n, n))

    do j = 1, n
        do i = 1, n
            A(i, j) = i + j
        end do
    end do

    call cpu_time(start)

    allocate(R(n, n), Q(n, n))
    Q = 0
    R = A

    call qr_reflect(R, Q, n)

    call cpu_time(end)
    write (*,*) "Time in seconds: "
    print *, end - start

    deallocate (A, R, Q)

END PROGRAM main

subroutine houshh_alt(x, n, h)
    implicit none
    
    integer, intent(in) :: n
    double precision, intent(in), dimension(n) :: x
    double precision, intent(out), dimension(n) :: h
    double precision, external :: dnrm2
    double precision :: alpha
    double precision :: u_n

    alpha = dnrm2(n, x, 1)

    h = x
    h(1) = h(1) - alpha
    
    u_n = dnrm2(n, h, 1)
    h = h / u_n

end subroutine houshh_alt


subroutine hhMulRight(u, k, n, Q, tmp)
    implicit none
    
    integer, intent(in) :: n, k
    double precision, intent(inout), dimension(n, n) :: Q
    double precision, intent(in), dimension(n) :: u
    double precision, intent(out), dimension(n) :: tmp

    tmp = 0

    call dgemv('n', n, n - k + 1, 1d0, Q(:, k:), n, u, 1, 0d0, tmp, 1)
    call dger(n, n - k + 1, -2d0, tmp, 1, u, 1, Q(:, k:), n)

end subroutine hhMulRight


subroutine hhMulLeft(u, k, n, Q, tmp)
    implicit none
    
    integer, intent(in) :: n, k
    double precision, intent(inout), dimension(n, n) :: Q
    double precision, intent(in), dimension(n) :: u
    double precision, intent(out), dimension(n) :: tmp

    tmp = 0

    call dgemv('t', n - k + 1, n, 1d0, Q(k:, :), n - k + 1, u, 1, 0d0, tmp, 1)
    call dger(n - k + 1, n, -2d0, u, 1, tmp, 1, Q(k:, :), n - k + 1)

end subroutine hhMulLeft



subroutine qr_reflect(R, Q, n)
    implicit none
    
    integer, intent(in) :: n
    double precision, intent(inout), dimension(n, n) :: R, Q
    integer :: i, j
    double precision, dimension (:), allocatable :: vec_i, h, tmp
    
    do i = 1, n
        Q(i, i) = 1    ! make eye
    end do

    allocate (vec_i(n), h(n), tmp(n))  
    
    do i = 1, n - 1
        call dcopy(n, R(:, i), 1, vec_i, 1)

        call houshh_alt(vec_i, n - i + 1, h)    
        call hhMulLeft(h, i, n, R, tmp)
        call hhMulRight(h, i, n, Q, tmp)
    end do

    deallocate (vec_i, h, tmp)

end subroutine qr_reflect
