use std::alloc::Layout;
use std::alloc::alloc;
use std::alloc::dealloc;
#[cfg(debug_assertions)]
use std::collections::HashSet;
use std::ptr::NonNull;
use std::ptr::null_mut;

pub struct LinkedList<T> {
    sentenial: *mut LinkedNode<T>,
    len: usize,

    #[cfg(debug_assertions)]
    debug_nodes: HashSet<*mut LinkedNode<T>>,
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        unsafe {
            let sentenial = alloc(Layout::new::<LinkedNode<T>>()) as *mut LinkedNode<T>;
            if sentenial.is_null() {
                panic!("unable to initialize linked list");
            }
            (*sentenial).prev = sentenial;
            (*sentenial).next = sentenial;
            Self {
                sentenial,
                len: 0,

                #[cfg(debug_assertions)]
                debug_nodes: HashSet::new(),
            }
        }
    }

    #[inline]
    pub fn push_front(&mut self, node: NonNull<LinkedNode<T>>) {
        unsafe {
            self.link_after(self.sentenial, node);
        }
    }

    #[inline]
    pub fn push_back(&mut self, node: NonNull<LinkedNode<T>>) {
        unsafe {
            self.link_before(self.sentenial, node);
        }
    }

    #[inline]
    pub fn pop_front(&mut self) -> Option<NonNull<LinkedNode<T>>> {
        unsafe {
            let node = (*self.sentenial).next;
            if node != self.sentenial {
                let node = NonNull::new_unchecked(node);
                self.detach(node);
                Some(node)
            } else {
                None
            }
        }
    }

    #[inline]
    pub fn pop_back(&mut self) -> Option<NonNull<LinkedNode<T>>> {
        unsafe {
            let node = (*self.sentenial).prev;
            if node != self.sentenial {
                let node = NonNull::new_unchecked(node);
                self.detach(node);
                Some(node)
            } else {
                None
            }
        }
    }

    #[inline]
    pub unsafe fn move_to_front(&mut self, node: NonNull<LinkedNode<T>>) {
        unsafe {
            if node.as_ref().prev == self.sentenial {
                return;
            }
            self.detach(node);
            self.link_after(self.sentenial, node);
        }
    }

    #[inline]
    pub unsafe fn move_to_back(&mut self, node: NonNull<LinkedNode<T>>) {
        unsafe {
            if node.as_ref().next == self.sentenial {
                return;
            }
            self.detach(node);
            self.link_before(self.sentenial, node);
        }
    }

    #[inline]
    pub unsafe fn detach(&mut self, node: NonNull<LinkedNode<T>>) {
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.debug_nodes.remove(&node.as_ptr()));
        }
        unsafe {
            let node_ptr = node.as_ptr();
            let prev_ptr = (*node_ptr).prev;
            let next_ptr = (*node_ptr).next;

            (*prev_ptr).next = next_ptr;
            (*next_ptr).prev = prev_ptr;
            (*node_ptr).prev = null_mut();
            (*node_ptr).next = null_mut();
        }
        self.len -= 1;
    }

    #[inline]
    unsafe fn link_after(&mut self, prev_ptr: *mut LinkedNode<T>, node: NonNull<LinkedNode<T>>) {
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                prev_ptr == self.sentenial || (self.debug_nodes.contains(&prev_ptr) && prev_ptr != node.as_ptr())
            );
            debug_assert!(self.debug_nodes.insert(node.as_ptr()));
        }
        unsafe {
            let next_ptr = (*prev_ptr).next;
            let node_ptr = node.as_ptr();

            (*prev_ptr).next = node_ptr;
            (*next_ptr).prev = node_ptr;
            (*node_ptr).prev = prev_ptr;
            (*node_ptr).next = next_ptr;
        }
        self.len += 1;
    }

    #[inline]
    unsafe fn link_before(&mut self, next_ptr: *mut LinkedNode<T>, node: NonNull<LinkedNode<T>>) {
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                next_ptr == self.sentenial || (self.debug_nodes.contains(&next_ptr) && next_ptr != node.as_ptr())
            );
            debug_assert!(self.debug_nodes.insert(node.as_ptr()));
        }
        unsafe {
            let prev_ptr = (*next_ptr).prev;
            let node_ptr = node.as_ptr();

            (*prev_ptr).next = node_ptr;
            (*next_ptr).prev = node_ptr;
            (*node_ptr).prev = prev_ptr;
            (*node_ptr).next = next_ptr;
        }
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        unsafe {
            debug_assert_eq!(self.sentenial, (*self.sentenial).prev);
            debug_assert_eq!(self.sentenial, (*self.sentenial).next);
            debug_assert_eq!(0, self.len());
        }
        unsafe {
            // clear attached nodes just in case
            let mut node = (*self.sentenial).next;
            while node != self.sentenial {
                let next = (*node).next;
                self.detach(NonNull::new_unchecked(node));
                node = next;
            }

            (*self.sentenial).next = null_mut();
            (*self.sentenial).prev = null_mut();
            dealloc(self.sentenial.cast::<u8>(), Layout::new::<LinkedNode<T>>());
        }
    }
}

impl<T> std::fmt::Debug for LinkedList<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("LinkedList");
        ds.field("len", &self.len);
        unsafe {
            let mut v: Vec<*mut LinkedNode<T>> = Vec::with_capacity(self.len);
            let mut p = (*self.sentenial).next;
            while p != self.sentenial && v.len() < self.len {
                v.push(p);
                p = (*p).next;
            }
            ds.field("list_nodes", &v);
        }
        #[cfg(debug_assertions)]
        {
            ds.field("debug_nodes", &self.debug_nodes.len());
        }
        ds.finish()
    }
}

#[derive(Debug)]
pub struct LinkedNode<T> {
    item: T,

    prev: *mut LinkedNode<T>,
    next: *mut LinkedNode<T>,
}

impl<T> LinkedNode<T> {
    #[inline]
    pub fn new(item: T) -> Box<Self> {
        Box::new(Self {
            item,
            prev: null_mut(),
            next: null_mut(),
        })
    }

    #[inline]
    pub fn is_attached(&self) -> bool {
        debug_assert_eq!(self.prev.is_null(), self.next.is_null());
        !self.prev.is_null() && !self.next.is_null()
    }

    #[inline]
    pub fn into_inner(self) -> T {
        debug_assert_eq!(null_mut(), self.prev);
        debug_assert_eq!(null_mut(), self.next);
        self.item
    }
}

impl<T> std::ops::Deref for LinkedNode<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

impl<T> std::ops::DerefMut for LinkedNode<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.item
    }
}

impl<T> AsRef<T> for LinkedNode<T> {
    fn as_ref(&self) -> &T {
        &self.item
    }
}

impl<T> AsMut<T> for LinkedNode<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.item
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Deref;
    use std::ptr::NonNull;

    use super::*;

    #[test]
    fn test_push_pop_front_back() {
        let mut list = LinkedList::<usize>::new();
        assert_eq!(list.len(), 0);

        let node0 = box_into_nn(LinkedNode::new(0));
        let node1 = box_into_nn(LinkedNode::new(1));
        let node2 = box_into_nn(LinkedNode::new(2));

        list.push_back(node0); // [0]
        list.push_back(node1); // [0, 1]
        list.push_front(node2); // [2, 0, 1]
        assert_eq!(3, list.len());

        let node = list.pop_front().unwrap(); // 2
        assert_eq!(2, *unsafe { node.as_ref() }.deref());
        nn_into_box(node);
        assert_eq!(2, list.len());

        let node = list.pop_front().unwrap(); // 0
        assert_eq!(0, *unsafe { node.as_ref() }.deref());
        nn_into_box(node);
        assert_eq!(1, list.len());

        let node = list.pop_back().unwrap(); // 1
        assert_eq!(1, *unsafe { node.as_ref() }.deref());
        nn_into_box(node);
        assert_eq!(0, list.len());

        assert!(list.pop_front().is_none());
        assert!(list.pop_back().is_none());
    }

    #[test]
    fn test_move_detach_front_back() {
        let mut list = LinkedList::<usize>::new();
        assert_eq!(list.len(), 0);

        let node0 = box_into_nn(LinkedNode::new(0));
        let node1 = box_into_nn(LinkedNode::new(1));
        let node2 = box_into_nn(LinkedNode::new(2));

        list.push_back(node0); // [0]
        list.push_back(node1); // [0, 1]
        list.push_front(node2); // [2, 0, 1]
        assert_eq!(3, list.len());

        unsafe {
            list.move_to_front(node1); // [1, 2, 0]
        }
        assert_eq!(3, list.len());

        unsafe {
            list.move_to_back(node2); // [1, 0, 2]
        }
        assert_eq!(3, list.len());

        unsafe {
            list.detach(node0); // [1, 2]
        }
        assert_eq!(2, list.len());

        unsafe {
            list.detach(node1); // [2]
        }
        assert_eq!(1, list.len());

        unsafe {
            list.detach(node2); // []
        }
        assert_eq!(0, list.len());

        assert!(list.pop_front().is_none());
        assert!(list.pop_back().is_none());
    }

    fn box_into_nn<T>(node: Box<LinkedNode<T>>) -> NonNull<LinkedNode<T>> {
        unsafe { NonNull::new_unchecked(Box::into_raw(node)) }
    }

    fn nn_into_box<T>(node: NonNull<LinkedNode<T>>) -> Box<LinkedNode<T>> {
        unsafe { Box::from_raw(node.as_ptr()) }
    }
}
