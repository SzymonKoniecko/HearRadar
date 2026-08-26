fn sum(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_both() {
        assert_eq!(sum(2, 3), 5);
    }

    #[test]
    fn divide_both() {
        assert_eq!(sum(-2, 3), 1);
    }
}
