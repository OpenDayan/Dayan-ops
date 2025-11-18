#ifndef _NPU_LIST_H_
#define _NPU_LIST_H_

#include <type_traits>

class NPUListNodeBase {
public:
    NPUListNodeBase* next;
    NPUListNodeBase** prev_next;

    NPUListNodeBase() : next(nullptr), prev_next(nullptr) {}
};


template<typename T>
class NPUListHead {
    static_assert(std::is_base_of<NPUListNodeBase, T>::value, "T must inherit from NPUListNodeBase");
public:
    NPUListNodeBase* first;
    NPUListNodeBase** last; // 指向最后一个节点的next指针的地址

    NPUListHead() noexcept;
    bool list_empty() const noexcept;

    void list_insert_tail(T* node) noexcept;
    T* list_remove_head() noexcept;
    void list_remove(T* node) noexcept;
    T* list_head() noexcept;
    T* list_next(T* node) noexcept;
};

template<typename T>
inline NPUListHead<T>::NPUListHead() noexcept {
    first = nullptr;
    last = &first;
}

template<typename T>
inline void NPUListHead<T>::list_insert_tail(T* p) noexcept {
    auto node = static_cast<NPUListNodeBase*>(p);
    node->next = nullptr;
    node->prev_next = last;
    *last = node;
    last = &node->next;
}

template<typename T>
inline T* NPUListHead<T>::list_remove_head() noexcept {
    if (first == nullptr) return nullptr;
    
    auto* item = first;
    list_remove(static_cast<T*>(item));
    return static_cast<T*>(item);
}

template<typename T>
inline void NPUListHead<T>::list_remove(T* p) noexcept {
    auto node = static_cast<NPUListNodeBase*>(p);
    if (node->next != nullptr) {
        node->next->prev_next = node->prev_next;
    } else {
        last = node->prev_next;
    }
    *node->prev_next = node->next;
    node->next = nullptr;
    node->prev_next = nullptr;
}

template<typename T>
inline T* NPUListHead<T>::list_head() noexcept {
    return static_cast<T*>(first);
}

template<typename T>
inline T* NPUListHead<T>::list_next(T* node) noexcept {
    return static_cast<T*>(node->next);
}

template<typename T>
inline bool NPUListHead<T>::list_empty() const noexcept {
    return first == nullptr;
}

#endif
