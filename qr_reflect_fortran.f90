PROGRAM main
    implicit none
 
    double precision, dimension (:, :), allocatable :: A, R, Q
    integer :: n, i, j
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


subroutine norm(v, n, res)
    implicit none
    
    integer, intent(in) :: n
    double precision, intent(in), dimension(n) :: v
    double precision, intent(out) :: res
    integer :: i

    res = 0
    do i = 1, n
        res = res + v(i)*v(i)
    end do
    res = sqrt(res)

end subroutine norm


subroutine houshh_alt(x, n, h)
    implicit none
    
    integer, intent(in) :: n
    double precision, intent(in), dimension(n) :: x
    double precision, intent(out), dimension(n) :: h
    double precision :: alpha
    double precision :: u_n

    call norm(x, n, alpha)

    h = x
    h(1) = h(1) - alpha
    
    call norm(h, n, u_n)
    h = h / u_n

end subroutine houshh_alt


subroutine hhMulRight(u, k, n, Q, tmp)
    implicit none
    
    integer, intent(in) :: n, k
    double precision, intent(inout), dimension(n, n) :: Q
    double precision, intent(in), dimension(n) :: u
    double precision, intent(out), dimension(n) :: tmp
    integer :: i, j

    
    tmp = 0

    do j = k, n
        do i = 1, n
            tmp(i) = tmp(i) + Q(i, j) * u(j - k + 1) 
        end do
    end do
    
    do j = k, n
        do i = 1, n
            Q(i, j) = Q(i, j) - 2*tmp(i) * u(j - k + 1) 
        end do
    end do

end subroutine hhMulRight


subroutine hhMulLeft(u, k, n, Q, tmp)
    implicit none
    
    integer, intent(in) :: n, k
    double precision, intent(inout), dimension(n, n) :: Q
    double precision, intent(in), dimension(n) :: u
    double precision, intent(out), dimension(n) :: tmp
    integer :: i, j

    
    tmp = 0

    do j = 1, n  
        do i = k, n     
            tmp(j) = tmp(j) + Q(i, j) * u(i - k + 1) 
        end do
    end do   
    
    do j = 1, n  
        do i = k, n   
            Q(i, j) = Q(i, j) - 2*tmp(j) * u(i - k + 1) 
        end do
    end do

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
        do j = i, n
            vec_i(j - i + 1) = R(j, i)
        end do

        call houshh_alt(vec_i, n - i + 1, h)   
        call hhMulLeft(h, i, n, R, tmp)
        call hhMulRight(h, i, n, Q, tmp)
    end do

    deallocate (vec_i, h, tmp)

end subroutine qr_reflect
